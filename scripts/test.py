import argparse
import json
import math
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "aerosol_data.xlsx")
DEFAULT_MODELS = os.path.join(ROOT, "models")

UM = "(\u03bcm)"
COLS = dict(cov="Spray Coverage", Q="Spray Flow Rate (L/min)", Dd="Droplet mean size  " + UM,
            orf="Orfice diameter (mm)", dm="Mist mean size " + UM, Cm="Mist mean concentration (P/cm^3)",
            rho="Aerosol density (kg/m3)", CA="Contact Angle", dp="Aerosol size " + UM)
COND = ["cov", "Q", "Dd", "orf", "dm", "Cm", "rho", "CA"]
TIME_COLS = ["0min", "1min", "3min", "5min", "10min", "15min", "20min", "30min", "40min", "50min", "60min"]
T_MIN = np.array([0, 1, 3, 5, 10, 15, 20, 30, 40, 50, 60], float)
T_H = T_MIN / 60.0
T_PRED = np.array([0, 1, 3, 5, 10, 20, 30, 40, 50, 60], float)
N_RUNS, N_CHANNELS = 35, 48

V_VESSEL = 3.92
H_VESSEL = 2.5
H_FALL = 2.3
SRP_ED_HIGH, SRP_ED_LOW, SRP_DF_SWITCH = 10.0, 1.0, 50.0
SRP_ACH = 2.0
SRP_COV_UNIFORM = 0.90

ZR_RUNS = list(range(1, 33))
NEW_RUNS = {33: "SiC", 34: "CeO2", 35: "TiO2"}
TEST_RUNS = [4, 5, 15, 16, 29, 30]
FOLD_RUNS = [[1, 2, 3, 13, 14], [8, 9, 10, 22, 23], [17, 18, 19, 27, 28], [20, 21, 24, 25, 26],
             [6, 7, 11, 12, 31, 32]]
DEV_RUNS = sorted(r for f in FOLD_RUNS for r in f)
SPLITS = {f"cv{k}": [r for r in DEV_RUNS if r not in f] for k, f in enumerate(FOLD_RUNS)}
SPLITS.update(test=list(DEV_RUNS), all32=list(ZR_RUNS))
CV_PACKS = [f"cv{k}" for k in range(5)]
PACK_SPLIT = dict(dev26="test", all32="all32", t30="test", **{p: p for p in CV_PACKS})
PACK_TMAX = dict(t30=30.0)
NOZZLES = {"N1": (309.0, 1.20), "N2": (344.0, 1.70), "N3": (302.0, 1.20), "N4": (162.0, 2.91)}
ZRO2 = (5600.0, 66.0)

KB = 1.380649e-23
T_AIR = 293.15
MU_AIR = 1.81e-5
RHO_AIR = 1.204
MFP = 0.0665e-6
RHO_W = 998.0
MU_W = 1.002e-3
G = 9.81
SIG_REL = 0.05
A_MAX = 10.0
V_MIST_LO, V_MIST_HI = 1.70664e5 * 0.71 ** 3, 2.70037e5 * 0.78 ** 3

Z_SEED = 123
SIG_FLOOR = 0.005
TAUS = np.round(np.concatenate([[0.025], np.arange(0.05, 0.951, 0.05), [0.975]]), 3)
CRPS_IDX = np.arange(1, 20)
ALPHAS = [0.2, 0.1, 0.05]
PLAN_ALPHAS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
CONF_LEVELS = sorted(round(1 - a, 2) for a in PLAN_ALPHAS)
Y_STAR = 0.1
Y_STARS = (0.2, 0.1, 0.05)
TF = np.linspace(0, 600, 6001)
GG_LO, GG_HI = 0.198, 1.0
ADM_TOL = 1e-3
AD_FEATURES = ["ln Q", "ln Dd", "ln orifice", "mist on", "mist on x ln Cm", "mist on x ln dm", "ln coverage"]
SCENARIO_COLS = ["id", "Q_lpm", "coverage", "mist_dm_um", "mist_Cm_cm3", "rho_kgm3", "contact_angle_deg"]

MISSING_DATA = ("The experimental data set of the paper is provided as data/aerosol_data.xlsx in this repository. "
                "Restore it there or pass a data file in the same layout with --data <file> (.xlsx or .csv), "
                "for example the synthetic example data/example/synthetic_aerosol_data.csv. "
                "Inference with the shipped models needs no data: python scripts/test.py --predict "
                "data/example/example_conditions.csv")


def load_data(path):
    if not os.path.exists(path):
        print(f"Data file not found: {path}\n{MISSING_DATA}")
        sys.exit(1)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, float_precision="round_trip")
    else:
        df = pd.read_excel(path, sheet_name="Sheet1")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    missing = [c for c in [COLS[k] for k in COND + ["dp"]] + TIME_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in {path}: {missing}")
    dpv = df[COLS["dp"]].values
    run = np.ones(len(df), int)
    for i in range(1, len(df)):
        run[i] = run[i - 1] + (1 if dpv[i] <= dpv[i - 1] else 0)
    if run.max() != N_RUNS:
        sys.exit(f"Expected {N_RUNS} runs (detected where the aerosol size sequence restarts), found {run.max()}")
    dp = np.array(sorted(df[COLS["dp"]].round(3).unique()))
    if len(dp) != N_CHANNELS:
        sys.exit(f"Expected {N_CHANNELS} aerosol size channels, found {len(dp)}")
    Y = np.full((N_RUNS, N_CHANNELS, len(T_MIN)), np.nan)
    cond = np.zeros((N_RUNS, len(COND)))
    for r in range(1, N_RUNS + 1):
        sub = df[run == r]
        c = sub[[COLS[k] for k in COND]].values
        if not (c == c[0]).all():
            sys.exit(f"Operating conditions vary inside run {r}")
        cond[r - 1] = c[0]
        j = np.searchsorted(dp, sub[COLS["dp"]].round(3).values)
        Y[r - 1, j] = sub[TIME_COLS].apply(pd.to_numeric, errors="coerce").values.astype(float)
    present = np.isfinite(Y[:, :, 0])
    Y[:, :, 0] = np.where(present, 1.0, np.nan)
    nozzle = np.array([nozzle_of(c[2], c[3]) for c in cond])
    return SimpleNamespace(cond=cond, Y=Y, mask=np.isfinite(Y), present=present, dp=dp, nozzle=nozzle)


def nozzle_of(Dd, orf):
    for k, (d, o) in NOZZLES.items():
        if abs(Dd - d) < 0.5 and abs(orf - o) < 1e-3:
            return k
    return "custom"


def cunningham(d):
    kn = 2.0 * MFP / d
    return 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))


def diffusivity(d):
    return KB * T_AIR * cunningham(d) / (3.0 * np.pi * MU_AIR * d)


def relaxation_time(d, rho):
    return rho * d ** 2 * cunningham(d) / (18.0 * MU_AIR)


def settling_velocity(d, rho):
    return relaxation_time(d, rho) * G


def droplet_terminal_velocity(D):
    D = np.asarray(D, float)
    U = np.full_like(D, 1.0)
    for _ in range(200):
        Re = np.maximum(RHO_AIR * U * D / MU_AIR, 1e-9)
        cd = 24.0 / Re * (1 + 0.15 * Re ** 0.687) + 0.42 / (1 + 42500.0 * Re ** -1.16)
        U_new = np.sqrt(4.0 * (RHO_W - RHO_AIR) * G * D / (3.0 * cd * RHO_AIR))
        if np.allclose(U_new, U, rtol=1e-10):
            break
        U = 0.5 * U + 0.5 * U_new
    return U


def slinn_efficiencies(dp, D, rho_p):
    dp, D, rho_p = np.broadcast_arrays(np.asarray(dp, float), np.asarray(D, float), np.asarray(rho_p, float))
    U = droplet_terminal_velocity(D)
    Re = D * U * RHO_AIR / (2.0 * MU_AIR)
    Sc = MU_AIR / (RHO_AIR * diffusivity(dp))
    ED = 4.0 / (Re * Sc) * (1 + 0.4 * Re ** 0.5 * Sc ** (1 / 3) + 0.16 * Re ** 0.5 * Sc ** 0.5)
    phi_r = dp / D
    omega = MU_W / MU_AIR
    ER = 4.0 * phi_r * (1.0 / omega + (1 + 2 * Re ** 0.5) * phi_r)
    tau = relaxation_time(dp, rho_p)
    St = 2.0 * tau * np.maximum(U - settling_velocity(dp, rho_p), 0.0) / D
    return ED, ER, St


def _fuchs_g(d, D_b, c):
    l = 8.0 * D_b / (np.pi * c)
    return ((d + l) ** 3 - (d ** 2 + l ** 2) ** 1.5) / (3.0 * d * l) - d


def fuchs_kernel(d1, rho1, d2, rho2=RHO_W):
    D1, D2 = diffusivity(d1), diffusivity(d2)
    m1, m2 = rho1 * np.pi * d1 ** 3 / 6.0, rho2 * np.pi * d2 ** 3 / 6.0
    c1, c2 = np.sqrt(8 * KB * T_AIR / (np.pi * m1)), np.sqrt(8 * KB * T_AIR / (np.pi * m2))
    g12 = np.sqrt(_fuchs_g(d1, D1, c1) ** 2 + _fuchs_g(d2, D2, c2) ** 2)
    c12 = np.sqrt(c1 ** 2 + c2 ** 2)
    s = d1 + d2
    beta = 1.0 / (s / (s + 2 * g12) + 8 * (D1 + D2) / (c12 * s))
    return 2 * np.pi * (D1 + D2) * s * beta


def spray_rate_per_efficiency(Q_lpm, D_um, V, h):
    F = Q_lpm / 1000.0 / 60.0
    return 3.0 * h * F / (2.0 * V * D_um * 1e-6) * 3600.0


def physics_tensors(cond, dp_um):
    dp = dp_um * 1e-6
    R, J = len(cond), len(dp)
    basis = np.zeros((R, J, 4))
    settle = np.zeros((R, J))
    S = np.zeros(R)
    for n in range(R):
        Dd, Q, rho = cond[n, 2], cond[n, 1], cond[n, 6]
        dm, Cm = cond[n, 4], cond[n, 5]
        ED, ER, St = slinn_efficiencies(dp, Dd * 1e-6, rho)
        basis[n, :, 0], basis[n, :, 1], basis[n, :, 2] = ED, ER, St / (1.0 + St)
        if Cm > 0 and dm > 0:
            basis[n, :, 3] = fuchs_kernel(dp, rho, dm * 1e-6) * Cm * 1e6 * 3600.0
        settle[n] = settling_velocity(dp, rho) / H_VESSEL * 3600.0
        S[n] = spray_rate_per_efficiency(Q, Dd, V_VESSEL, H_FALL)
    return basis, settle, S


def raw_mist(c):
    return np.column_stack([np.log(np.maximum(c[:, 4], 1e-3)), np.log(np.maximum(c[:, 5], 1.0))])


def mist_amount(c):
    on = ((c[:, 4] > 0) & (c[:, 5] > 0)).astype(float)
    v = np.maximum(c[:, 5], 1.0) * np.maximum(c[:, 4], 1e-3) ** 3
    xi = np.clip((np.log(v) - np.log(V_MIST_LO)) / (np.log(V_MIST_HI) - np.log(V_MIST_LO)), 0.0, 2.0) * on
    return np.column_stack([on, xi])


def make_batch(cond, dp_um, prep, times):
    basis, settle, S = physics_tensors(cond, dp_um)
    on = ((cond[:, 4] > 0) & (cond[:, 5] > 0)).astype(float)
    spray = (np.column_stack([np.log(cond[:, 2]), np.log(cond[:, 3])]) - prep["mu_s"]) / prep["sd_s"]
    flow = (np.log(cond[:, 1]) - prep["mu_q"])[:, None]
    mist = (raw_mist(cond) - prep["mu_m"]) / prep["sd_m"] * on[:, None]
    mat = np.column_stack([(cond[:, 6] - 5600.0) / 5600.0, (cond[:, 7] - 66.0) / 66.0])
    lcov = (np.log(cond[:, 0]) - prep["mu_c"])[:, None]
    ldp = np.log(dp_um)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return dict(spray=f(spray), flow=f(flow), mist=f(mist), on=f(on), mat=f(mat), mistx=f(mist_amount(cond)),
                lcov=f(lcov), basis=f(basis), settle=f(settle), S=f(S), t=f(times),
                logdp=f((ldp - ldp.mean()) / ldp.std()))


class CondTokens(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.spray = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.mist = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.mist_off = nn.Parameter(torch.zeros(d))
        self.mat_w = nn.Parameter(torch.zeros(2, d))
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)

    def forward(self, B):
        on = B["on"].unsqueeze(-1) * 0.0
        mist = on * self.mist(B["mist"]) + (1 - on) * self.mist_off
        mat = B["mat"] @ self.mat_w
        tok = torch.stack([self.spray(B["spray"]), mist, mat], 1)
        return tok + self.type_emb


class SizeTokens(nn.Module):
    def __init__(self, d, n_freq=8):
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n_freq + 1).float())
        self.proj = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, logdp, R):
        x = logdp.view(-1, 1)
        f = torch.cat([x, torch.sin(x * self.freq), torch.cos(x * self.freq)], -1)
        return self.proj(f).unsqueeze(0).expand(R, -1, -1)


class Encoder(nn.Module):
    def __init__(self, d, heads, n_cond, n_spec, dropout):
        super().__init__()
        self.cond = CondTokens(d)
        self.size = SizeTokens(d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.cond_enc = nn.TransformerEncoder(el, n_cond, enable_nested_tensor=False)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.spec = nn.TransformerDecoder(dl, n_spec)
        self.norm_c = nn.LayerNorm(d)
        self.norm_s = nn.LayerNorm(d)

    def forward(self, B):
        R = B["spray"].shape[0]
        c = self.cond_enc(self.cond(B))
        pooled = self.norm_c(c.mean(1))
        h = self.spec(self.size(B["logdp"], R), c)
        return pooled, self.norm_s(h)


class TraceFormer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        d = hp["d"]
        self.enc = Encoder(d, hp["heads"], hp["n_cond"], hp["n_spec"], hp["dropout"])
        self.wsu_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.25))))
        self.gam_head = nn.Sequential(nn.Linear(17, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.gam_head[-1].weight)
        with torch.no_grad():
            self.gam_head[-1].bias.fill_(math.log(math.expm1(0.1)))
        self.register_buffer("gfreq", torch.arange(1, 9).float())
        self.covm_raw = nn.Parameter(torch.full((8,), math.log(math.expm1(0.3))))
        self.register_buffer("mono_c", torch.tensor([0., 1., 0., 1., 1., 1., 1., 0.]))
        self.mist_raw = nn.Parameter(torch.full((2, 8), -2.5))
        self.dz = hp["dz"]
        self.run_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 16))
        nn.init.normal_(self.run_head[-1].weight, std=0.01)
        self.register_buffer("mono", torch.tensor([1., 1., 0., 1., 1., 1., 1., 1.]))
        self.register_buffer("mono_m", torch.tensor([1., 1., 0., 1., 1., 1., 1., 1.]))
        b0 = torch.tensor([math.log(0.37 / 0.63), math.log(0.35), math.log(0.02), math.log(0.19 / 0.81),
                           math.log(0.1 / 0.9), math.log(0.1 / 0.9), math.log(1.25), math.log(1.0)])
        bi = hp["beta_init"]
        with torch.no_grad():
            self.run_head[-1].bias.copy_(torch.cat([b0, torch.log(torch.expm1(torch.tensor([bi] * 8).clamp_min(1e-3)))]))
        self.beta_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor([bi, bi, 1e-3, bi, bi, bi, hp["beta_aI_init"], bi]))))
        self.size_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.size_head[-1].weight)
        with torch.no_grad():
            self.size_head[-1].bias.copy_(torch.tensor([0.0, math.log(math.expm1(0.1))]))
        self.w_z = nn.Parameter(torch.zeros(4, self.dz))
        nn.init.normal_(self.w_z, std=0.05)
        self.log_sig_floor = nn.Parameter(torch.tensor(math.log(0.006)))

    def parts(self, B, z, enc):
        pooled, h = enc
        uu = self.run_head(pooled)
        beta = (F.softplus(self.beta_raw) * self.mono).unsqueeze(0).expand(uu.shape[0], -1)
        u = uu[:, :8] + beta * B["flow"]
        u = u + B["mistx"] @ (F.softplus(self.mist_raw) * self.mono_m)
        u = u + B["lcov"] * (F.softplus(self.covm_raw) * self.mono_c).unsqueeze(0)
        m = z @ self.w_z.t()
        dphi, dk, de, dl = m[:, 0:1], m[:, 1:2], m[:, 2:3], m[:, 3:4]
        phi = torch.sigmoid(u[:, 0:1] + dphi).clamp(1e-3, 1 - 1e-3)
        k = torch.exp(u[:, 1:2] + dk)
        eps = 0.5 * torch.sigmoid(u[:, 3:4] + de)
        lam_w = torch.exp(u[:, 2:3])
        amp = torch.cat([A_MAX * torch.sigmoid(u[:, 4:6]), torch.exp(u[:, 6:8])], 1)
        lam0 = lam_w + B["settle"]
        phib = phi + (1 - phi) * eps
        sc = torch.exp(dl)
        so = self.size_head(h)
        rho0 = so[..., 0]
        x = B["logdp"].view(-1, 1)
        fx = torch.cat([x, torch.sin(x * self.gfreq), torch.cos(x * self.gfreq)], -1)
        gam = F.softplus(self.gam_head(fx)).view(1, -1).expand(so.shape[0], -1)
        rho = rho0 + gam * B["flow"]
        Sb = B["S"].unsqueeze(-1) / phib
        chan = torch.stack([Sb * amp[:, 0:1] * B["basis"][..., 0], Sb * amp[:, 1:2] * B["basis"][..., 1],
                            Sb * amp[:, 2:3] * B["basis"][..., 2],
                            amp[:, 3:4] * B["on"].unsqueeze(-1) * B["basis"][..., 3]], -1)
        corr = torch.exp(rho) * sc
        lam_in = chan.sum(-1) * corr
        return dict(phi=phi, k=k, eps=eps, phib=phib, lam0=lam0, lam_in=lam_in)

    def forward(self, B, z, enc):
        t = B["t"]
        P = self.parts(B, z, enc)
        th = 2.0 / 60.0
        w = F.softplus(self.wsu_raw)
        tm = (t + w * th * (1 - torch.exp(-t / th))).view(1, 1, -1)
        phi, k, eps, lam_in, lam0 = P["phi"], P["k"], P["eps"], P["lam_in"], P["lam0"]
        b = k * (1 - phi) / phi
        a11 = lam_in + b
        a22 = eps * lam_in + k
        disc = torch.sqrt((a11 - a22) ** 2 + 4 * k * b + 1e-12)
        tr = a11 + a22
        det = lam_in * (eps * lam_in + k + eps * b)
        mu1 = 0.5 * (tr + disc)
        mu2 = 2 * det / (tr + disc)
        h0 = (phi + (1 - phi) * eps) * lam_in
        a = ((h0 - mu2) / disc).clamp(0.0, 1.0)
        slow = torch.exp(-mu2.unsqueeze(-1) * tm)
        y = torch.exp(-lam0.unsqueeze(-1) * t.view(1, 1, -1)) * (a.unsqueeze(-1) * torch.exp(-mu1.unsqueeze(-1) * tm)
                                                                 + (1 - a).unsqueeze(-1) * slow)
        P.update(mu2=mu2 + lam0)
        return y, P


def load_group(models_dir, name):
    path = os.path.join(models_dir, f"trace_former_{name}.pt")
    if not os.path.exists(path):
        sys.exit(f"Missing model file {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("format") != "trace-former-v1":
        sys.exit(f"{path} is not a trace-former-v1 file")
    if obj["hp"].get("tmax") != PACK_TMAX.get(name):
        sys.exit(f"{path} was trained with the training window tmax = {obj['hp'].get('tmax')} min, "
                 f"expected {PACK_TMAX.get(name)}")
    if not obj["members"] or any(m["split"] != PACK_SPLIT[name] for m in obj["members"]):
        sys.exit(f"{path} must hold members of split {PACK_SPLIT[name]}")
    return obj["hp"], sorted(obj["members"], key=lambda m: m["seed"])


def build_member(hp, m):
    model = TraceFormer(hp)
    model.load_state_dict(m["state"])
    model.eval()
    prep = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in m["prep"].items()}
    return model, prep


@torch.no_grad()
def predict_samples(model, prep, cond, dp_um, n_z, times=T_H):
    B = make_batch(cond, dp_um, prep, times)
    enc = model.enc(B)
    g = torch.Generator().manual_seed(Z_SEED)
    Z = [torch.randn(len(cond), model.dz, generator=g) for _ in range(n_z)]
    ys, ps = [], []
    for z in Z:
        y, P = model(B, z, enc)
        ys.append(y)
        ps.append(P)
    par = {k: torch.stack([p[k] for p in ps]).mean(0).numpy() for k in ("phi", "k", "eps", "phib", "lam_in", "lam0", "mu2")}
    return torch.stack(ys).numpy(), par


def count_params(model):
    return int(sum(p.numel() for n, p in model.named_parameters() if n != "log_sig_floor"))


def metrics(Y, Yh, mask, floor=1e-3):
    y, p = Y[mask], Yh[mask]
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    wmape = float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)
    lr = float(np.sqrt(np.mean((np.log10(np.maximum(y, floor)) - np.log10(np.maximum(p, floor))) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(N=int(mask.sum()), RMSE=rmse, WMAPE=wmape, logRMSE=lr, R2=r2)


def eval_mask(D, runs, tmin=0.0, tmax=60.0):
    m = np.zeros_like(D.mask)
    idx = np.asarray(runs) - 1
    m[idx] = D.mask[idx]
    m[:, :, (T_MIN <= tmin) | (T_MIN > tmax)] = False
    m[:, :, 0] = False
    return m


def agg(vals):
    v = np.array([x for x in vals if x is not None])
    return (float(v.mean()), float(v.std(ddof=0)), len(v)) if len(v) else (np.nan, np.nan, 0)


def admissibility(A, tol=ADM_TOL):
    viol_range = np.any((A < -tol) | (A > 1 + tol), -1)
    viol_mono = np.any(np.diff(A, axis=-1) > tol, -1)
    viol_init = np.abs(A[..., 0] - 1) > tol
    return dict(range=float(viol_range.mean() * 100), monotone=float(viol_mono.mean() * 100),
                initial=float(viol_init.mean() * 100), any=float((viol_range | viol_mono | viol_init).mean() * 100))


def oof(means, seed):
    P = np.full((N_RUNS, N_CHANNELS, len(T_MIN)), np.nan)
    for k, runs in enumerate(FOLD_RUNS):
        idx = np.asarray(runs) - 1
        P[idx] = means[f"cv{k}"][seed][idx]
    return P


def table5(D, means, seeds, n_params, means30):
    Y = np.nan_to_num(D.Y)
    m_dev = eval_mask(D, DEV_RUNS)
    m_test = eval_mask(D, TEST_RUNS)
    cv, te, adm = [], [], []
    for s in seeds:
        cv.append(metrics(Y, oof(means, s), m_dev))
        Pt = means["test"][s]
        te.append(metrics(Y, Pt, m_test))
        adm.append(admissibility(Pt))
    ex = [metrics(Y, P, eval_mask(D, TEST_RUNS, tmin=30)) for P in means30]
    exd = [metrics(Y, P, eval_mask(D, DEV_RUNS, tmin=30)) for P in means30]
    rec = dict(model="TRACE-Former")
    for name, L in (("cv", cv), ("test", te), ("extrap_test", ex), ("extrap_dev", exd)):
        for met in ("RMSE", "WMAPE", "logRMSE", "R2"):
            mu, sd, n = agg([x[met] for x in L])
            rec[f"{name}_{met}"], rec[f"{name}_{met}_sd"], rec[f"{name}_n"] = mu, sd, n
    for key in ("range", "monotone", "initial", "any"):
        rec[f"viol_{key}"] = agg([a[key] for a in adm])[0]
    rec["n_params"] = n_params
    s5 = seeds[:5]
    rec["cv_RMSE_ens5"] = metrics(Y, np.mean([oof(means, s) for s in s5], 0), m_dev)["RMSE"]
    e = metrics(Y, np.mean([means["test"][s] for s in s5], 0), m_test)
    rec["test_RMSE_ens5"], rec["test_logRMSE_ens5"] = e["RMSE"], e["logRMSE"]
    return rec


def ti(tau):
    return int(np.argmin(np.abs(TAUS - tau)))


def add_noise(S, seed=0):
    rng = np.random.default_rng(seed)
    sig = np.sqrt((SIG_REL * S) ** 2 + SIG_FLOOR ** 2)
    return np.maximum(S + sig * rng.standard_normal(S.shape).astype(np.float32), 0.0)


def wquantile(vals, w, q):
    o = np.argsort(vals, axis=-1)
    v = np.take_along_axis(vals, o, -1)
    cw = np.cumsum(w[o], axis=-1)
    idx = np.argmax(cw >= q - 1e-12, axis=-1)
    return np.take_along_axis(v, idx[..., None], -1)[..., 0]


def rearrange(L, U, mask):
    L, U = L.copy(), U.copy()
    for idx in zip(*np.nonzero(mask.any(-1))):
        m = mask[idx]
        L[idx][m] = -np.sort(-L[idx][m])
        U[idx][m] = -np.sort(-U[idx][m])
    return L, U


def crps_quantiles(Q, y):
    Q = Q[CRPS_IDX]
    tau = TAUS[CRPS_IDX].reshape(-1, *([1] * y.ndim))
    d = y[None] - Q
    qs = np.maximum(tau * d, (tau - 1) * d)
    return 2.0 * qs.mean(0)


def interval_metrics(L, U, y, alpha):
    cov = (y >= L) & (y <= U)
    w = U - L
    isc = w + 2 / alpha * (L - y) * (y < L) + 2 / alpha * (y - U) * (y > U)
    return dict(coverage=float(cov.mean()), width=float(w.mean()), interval_score=float(isc.mean()))


def samples32(S16):
    return np.concatenate([s.astype(np.float32) for s in S16], 0)


def q_trace(S16):
    return np.quantile(add_noise(samples32(S16), seed=1), TAUS, axis=0).astype(np.float32)


def calib_points(D):
    pts = []
    for k, runs in enumerate(FOLD_RUNS):
        m = eval_mask(D, runs)
        for (ri, j, kk) in zip(*np.nonzero(m)):
            pts.append((k, ri, j, kk))
    P = np.array(pts)
    _, inv, cnt = np.unique(P[:, 1], return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return P, w / w.sum(), len(cnt)


def cvplus_scores(Qf, Y, P):
    med_i, lo, hi = ti(0.5), ti(0.15), ti(0.85)
    c = np.array([Qf[k][med_i, r, j, kk] for (k, r, j, kk) in P])
    s = np.array([max(0.5 * (Qf[k][hi, r, j, kk] - Qf[k][lo, r, j, kk]), 1e-4) for (k, r, j, kk) in P])
    return np.abs(Y[P[:, 1], P[:, 2], P[:, 3]] - c) / s


def cvplus_centers(Q, idx):
    c = Q[ti(0.5)][idx]
    s = np.maximum(0.5 * (Q[ti(0.85)][idx] - Q[ti(0.15)][idx]), 1e-4)
    return c, s


def cvplus_bounds(C, S, Rs, Ws, alpha, n_runs):
    lev = min((1 - alpha) * (1 + 1.0 / n_runs), 1.0)
    n = len(C[0])
    L, U = np.zeros(n), np.zeros(n)
    ww = np.concatenate(Ws)
    for b0 in range(0, n, 256):
        sl = slice(b0, b0 + 256)
        lo = np.concatenate([C[k][sl, None] - Rs[k][None, :] * S[k][sl, None] for k in range(len(C))], 1)
        up = np.concatenate([C[k][sl, None] + Rs[k][None, :] * S[k][sl, None] for k in range(len(C))], 1)
        L[sl] = -wquantile(-lo, ww, lev)
        U[sl] = wquantile(up, ww, lev)
    return L, U


def to_traj(vals, idx, shape, fill):
    A = np.full(shape, fill, np.float32)
    A[idx] = vals
    return A


def uq_test(D, Qf, Qt, calib, n_members):
    Y = np.nan_to_num(D.Y)
    mt = eval_mask(D, TEST_RUNS)
    idx = np.nonzero(mt)
    yv = Y[idx]

    def assess(Qtest, conformal):
        r = {}
        for a in ALPHAS:
            if conformal is not None:
                L, U = conformal[a]
            else:
                L, U = Qtest[ti(a / 2)][idx], Qtest[ti(1 - a / 2)][idx]
            Lt, Ut = rearrange(np.clip(to_traj(L, idx, Y.shape, np.nan), 0, 1),
                               np.clip(to_traj(U, idx, Y.shape, np.nan), 0, 1), mask=mt)
            r[f"{int(round((1 - a) * 100))}"] = interval_metrics(Lt[idx], Ut[idx], yv, a)
        r["CRPS"] = float(crps_quantiles(Qtest[:, idx[0], idx[1], idx[2]], yv).mean())
        return r

    res = {"TRACE-Former (raw)": dict(assess(Qt, None), members=n_members)}
    CS = [cvplus_centers(Qf[k], idx) for k in range(5)]
    C, S = [c for c, _ in CS], [s for _, s in CS]
    alist = sorted({round(1 - 2 * abs(t - 0.5), 3) for t in TAUS if t < 0.5})
    confQ = {a: cvplus_bounds(C, S, calib["R"], calib["W"], a, calib["n_runs"]) for a in alist}
    Qc = np.zeros((len(TAUS), len(yv)), np.float32)
    for i, t in enumerate(TAUS):
        if abs(t - 0.5) < 1e-9:
            Qc[i] = Qt[i][idx]
        else:
            a_ = round(1 - 2 * abs(t - 0.5), 3)
            Qc[i] = confQ[a_][0] if t < 0.5 else confQ[a_][1]
    QcFull = np.zeros_like(Qt)
    QcFull[:, idx[0], idx[1], idx[2]] = np.sort(Qc, 0)
    res["TRACE-Former (conformal)"] = assess(QcFull, {a: confQ[a] for a in ALPHAS})
    return res


def uq_dev(D, Qf):
    Y = np.nan_to_num(D.Y)
    runs = DEV_RUNS
    md = eval_mask(D, runs)
    Q = np.full_like(Qf[0], np.nan)
    for k, fr in enumerate(FOLD_RUNS):
        idx = np.asarray(fr) - 1
        Q[:, idx] = Qf[k][:, idx]
    s_all = np.maximum(0.5 * (Q[ti(0.85)] - Q[ti(0.15)]), 1e-4)
    sc = {rr: (np.abs(Y - Q[ti(0.5)]) / s_all)[eval_mask(D, [rr])] for rr in runs}

    def loro(rr, a):
        others = [q for q in runs if q != rr]
        vals = np.concatenate([sc[q] for q in others])
        w = np.concatenate([np.full(len(sc[q]), 1.0 / len(sc[q])) for q in others])
        w /= w.sum()
        return wquantile(vals[None, :], w, min((1 - a) * (1 + 1.0 / len(others)), 1.0))[0]

    res = {}
    for name, mode in (("TRACE-Former (raw)", "raw"), ("TRACE-Former (conformal)", "norm")):
        r = {}
        for a in ALPHAS:
            il, ih = ti(a / 2), ti(1 - a / 2)
            if mode == "raw":
                L, U = Q[il].copy(), Q[ih].copy()
            else:
                L, U = np.full(Y.shape, np.nan), np.full(Y.shape, np.nan)
                for rr in runs:
                    qh = loro(rr, a)
                    i = rr - 1
                    s = np.maximum(0.5 * (Q[ti(0.85), i] - Q[ti(0.15), i]), 1e-4)
                    L[i], U[i] = Q[ti(0.5), i] - qh * s, Q[ti(0.5), i] + qh * s
            L, U = rearrange(np.clip(L, 0, 1), np.clip(U, 0, 1), mask=md)
            r[f"{int(round((1 - a) * 100))}"] = interval_metrics(L[md], U[md], Y[md], a)
        if mode == "norm":
            Qc = np.full_like(Q, np.nan)
            Qc[ti(0.5)] = Q[ti(0.5)]
            for rr in runs:
                i = rr - 1
                s_ = np.maximum(0.5 * (Q[ti(0.85), i] - Q[ti(0.15), i]), 1e-4)
                for t_i, tau in enumerate(TAUS):
                    if abs(tau - 0.5) < 1e-9:
                        continue
                    qh = loro(rr, 1 - 2 * abs(tau - 0.5))
                    Qc[t_i, i] = Q[ti(0.5), i] + np.sign(tau - 0.5) * qh * s_
            Qc = np.sort(np.clip(Qc, 0, 1), 0)
            r["CRPS"] = float(crps_quantiles(Qc[:, md], Y[md]).mean())
        else:
            r["CRPS"] = float(crps_quantiles(Q[:, md], Y[md]).mean())
        res[name] = r
    return res


def loglin(tk, vals, tf=TF):
    v = np.log(np.clip(vals, 1e-6, 1.0))
    out = np.interp(tf, tk, v)
    if len(tk) >= 2:
        slope = min((v[-1] - v[-2]) / (tk[-1] - tk[-2]), 0.0)
        beyond = tf > tk[-1]
        out[beyond] = v[-1] + slope * (tf[beyond] - tk[-1])
    return np.exp(out)


def first_cross(curve, ys, tf=TF):
    b = curve <= ys
    return float(tf[np.argmax(b)]) if b.any() else np.inf


def gg_channels(D, r):
    return D.present[r - 1] & (D.dp >= GG_LO) & (D.dp <= GG_HI)


def measured_m(D, r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmax(D.Y[r - 1][gg_channels(D, r)], 0)
    k = np.isfinite(m)
    return T_MIN[k], m[k]


def oof_split(r):
    for k, f in enumerate(FOLD_RUNS):
        if r in f:
            return f"cv{k}"
    return "test"


def ad_features(c):
    on = ((c[:, 4] > 0) & (c[:, 5] > 0)).astype(float)
    return np.column_stack([np.log(c[:, 1]), np.log(c[:, 2]), np.log(c[:, 3]), on,
                            on * np.log(np.maximum(c[:, 5], 1)), on * np.log(np.maximum(c[:, 4], 1e-3)),
                            np.log(c[:, 0])])


def ad_distance(D):
    Fz = ad_features(D.cond[np.asarray(ZR_RUNS) - 1])
    mu, sd = Fz.mean(0), np.maximum(Fz.std(0), 1e-9)
    Fs = (Fz - mu) / sd
    dist = {r: float(np.min(np.linalg.norm(Fs[np.asarray(SPLITS[oof_split(r)]) - 1] - Fs[r - 1], axis=1)))
            for r in ZR_RUNS}
    dv = Fs[np.asarray(DEV_RUNS) - 1]
    nn_ = [np.min(np.linalg.norm(np.delete(dv, i, 0) - dv[i], axis=1)) for i in range(len(dv))]
    return dist, float(np.percentile(nn_, 95)), mu, sd, dv


def calib_scores(D, S16):
    sc = {}
    for k, runs in enumerate(FOLD_RUNS):
        Sn = add_noise(samples32(S16[f"cv{k}"]), seed=7)
        for r in runs:
            g = gg_channels(D, r)
            m = Sn[:, r - 1][:, g].max(1)
            kk = np.isfinite(D.Y[r - 1][g]).all(0) & (T_MIN > 0)
            q16, q50, q84 = np.quantile(m[:, kk], [0.16, 0.5, 0.84], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mm = np.nanmax(D.Y[r - 1][g], 0)[kk]
            sc[r] = (mm - q50) / np.maximum(0.5 * (q84 - q16), 1e-4)
    return sc


def plan_qhat(scores, alpha, exclude=None):
    others = [q for q in DEV_RUNS if q != exclude]
    vals = np.concatenate([scores[q] for q in others])
    w = np.concatenate([np.full(len(scores[q]), 1.0 / len(scores[q])) for q in others])
    w = w / w.sum()
    return wquantile(vals[None, :], w, min((1 - alpha) * (1 + 1.0 / len(others)), 1.0))[0]


def plan_bound(tk, m, qhat):
    q16, q50, q84 = np.quantile(m, [0.16, 0.5, 0.84], axis=0)
    U = np.minimum(q50 + qhat * np.maximum(0.5 * (q84 - q16), 1e-4), 1.0)
    U[tk == 0] = 1.0
    return np.minimum.accumulate(U)


def plan_raw_bound(tk, m, alpha):
    U = np.minimum(np.quantile(m, 1 - alpha, axis=0), 1.0)
    U[tk == 0] = 1.0
    return np.minimum.accumulate(U)


def srp_rate(Q_lpm, ed):
    F = Q_lpm * 60.0 / 1000.0
    return 3.0 * H_FALL * F * ed / (2.0 * V_VESSEL)


def expm2(M):
    s = int(max(0, np.ceil(np.log2(np.abs(M).sum(0).max() / 0.5))))
    X = M / 2.0 ** s
    E = np.eye(2)
    for k in range(18, 0, -1):
        E = np.eye(2) + X @ E / k
    for _ in range(s):
        E = E @ E
    return E


def srp_curve(cov, Q, t_h, n_sub=2000):
    single = cov >= SRP_COV_UNIFORM
    phi = 1.0 if single else cov
    tg = np.linspace(0, max(t_h.max(), 1e-6), n_sub + 1)
    dt = tg[1] - tg[0]
    E = []
    for lam in (srp_rate(Q, SRP_ED_HIGH), srp_rate(Q, SRP_ED_LOW)):
        if single:
            A = np.array([[lam, 0.0], [0.0, lam]])
        else:
            b = SRP_ACH * (1 - phi) / phi
            A = np.array([[lam / phi + b, -b], [-SRP_ACH, SRP_ACH]])
        E.append(expm2(-A * dt))
    c = np.array([1.0, 1.0])
    ys = [1.0]
    for _ in range(n_sub):
        y = phi * c[0] + (1 - phi) * c[1]
        c = E[0 if y > 1.0 / SRP_DF_SWITCH else 1] @ c
        ys.append(phi * c[0] + (1 - phi) * c[1])
    return np.interp(t_h, tg, np.array(ys))


def srp_time(cov, Q, ys):
    return first_cross(srp_curve(cov, Q, TF / 60.0), ys)


def planning(D, S16, scores, y_star):
    rows = []
    for r in ZR_RUNS:
        i = r - 1
        Sn = add_noise(np.concatenate([x[:, i].astype(np.float32) for x in S16[oof_split(r)]], 0), seed=r)
        g = gg_channels(D, r)
        kk = np.isfinite(D.Y[i][g]).all(0)
        tk, msamp = T_MIN[kk], Sn[:, g].max(1)[:, kk]
        tm, mm = measured_m(D, r)
        mcurve = loglin(tm, np.minimum.accumulate(mm))
        t_true = first_cross(mcurve[TF <= 60], y_star, TF[TF <= 60])
        Q = D.cond[i, 1]

        def outcome(ts):
            if not np.isfinite(ts):
                return dict(stop=False, verifiable=np.isfinite(t_true), safe=True, safe_ext=True, water=np.nan)
            ok = bool(np.interp(ts, TF, mcurve) <= y_star * (1 + 1e-9))
            if ts <= 60.0:
                return dict(stop=True, verifiable=True, safe=ok, safe_ext=ok, water=Q * ts)
            return dict(stop=False, verifiable=bool(np.isfinite(t_true)), safe=True, safe_ext=ok, water=Q * ts)
        for a in PLAN_ALPHAS:
            U = plan_bound(tk, msamp, plan_qhat(scores, a, exclude=r))
            rows.append(dict(method="TRACE-Former", alpha=a, Q=Q, t_true=t_true,
                             **outcome(first_cross(loglin(tk, U), y_star))))
            rows.append(dict(method="TRACE-Former (uncalibrated)", alpha=a, Q=Q, t_true=t_true,
                             **outcome(first_cross(loglin(tk, plan_raw_bound(tk, msamp, a)), y_star))))
        rows.append(dict(method="SRP 6.5.2", alpha=np.nan, Q=Q, t_true=t_true,
                         **outcome(srp_time(D.cond[i, 0], Q, y_star))))
        rows.append(dict(method="Oracle (measured)", alpha=np.nan, Q=Q, t_true=t_true, stop=bool(np.isfinite(t_true)),
                         verifiable=bool(np.isfinite(t_true)), safe=True, safe_ext=True,
                         water=Q * t_true if np.isfinite(t_true) else np.nan))
    return rows


def summarize(rows):
    T = pd.DataFrame(rows)
    out = []
    for (mth, a), g in T.groupby(["method", "alpha"], dropna=False):
        v = g[g.verifiable.astype(bool)]
        reach = g[np.isfinite(g.t_true)]
        out.append(dict(method=mth, alpha=a, n=len(g), n_verifiable=len(v),
                        reliability=float(v.safe.mean()) if len(v) else np.nan,
                        n_unsafe=int((~v.safe.astype(bool)).sum()), stop_rate=float(g.stop.mean()),
                        reliability_ext=float(g.safe_ext.astype(bool).mean()),
                        n_unsafe_ext=int((~g.safe_ext.astype(bool)).sum()),
                        water_reachable_median=float(reach.water.fillna(np.inf).median()) if len(reach) else np.nan,
                        water_reachable=float(reach.water.mean()) if len(reach) else np.nan,
                        water_reachable_oracle_ratio=float((reach.water.fillna(np.inf) / (reach.Q * reach.t_true)).median())
                        if len(reach) else np.nan))
    return pd.DataFrame(out)


def zeroshot(D, means32, train_rmse):
    Y = np.nan_to_num(D.Y)
    seeds = sorted(means32)
    med = np.nanmedian([train_rmse[s] for s in seeds])
    P = [means32[s] for s in seeds if not (np.isfinite(train_rmse[s]) and train_rmse[s] > 3 * med)]
    Pm = np.mean(P, 0)
    rows = []
    for r in NEW_RUNS:
        m = eval_mask(D, [r], tmax=20)
        met = metrics(Y, Pm, m)
        rows.append(dict(model="TRACE-Former", run=r, material=NEW_RUNS[r], RMSE=met["RMSE"],
                         RMSE_sd=float(np.std([metrics(Y, p, m)["RMSE"] for p in P])), logRMSE=met["logRMSE"],
                         WMAPE=met["WMAPE"], n_seeds=len(P)))
    m_all = eval_mask(D, list(NEW_RUNS), tmax=20)
    met = metrics(Y, Pm, m_all)
    rows.append(dict(model="TRACE-Former", run=0, material="all", RMSE=met["RMSE"],
                     RMSE_sd=float(np.std([metrics(Y, p, m_all)["RMSE"] for p in P])), logRMSE=met["logRMSE"],
                     WMAPE=met["WMAPE"], n_seeds=len(P)))
    return rows, Pm


def rate20(y, t):
    m = np.isfinite(y) & (y > 2e-3) & (t > 0)
    return -np.sum(np.log(y[m]) * t[m]) / np.sum(t[m] ** 2) if m.sum() >= 2 else np.nan


def rate_ratios(D, Pm):
    t = T_MIN / 60
    K = [k for k in range(len(t)) if T_MIN[k] <= 20]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in NEW_RUNS:
            obs, prd = [], []
            for j in range(N_CHANNELS):
                if not (D.present[r - 1, j] and D.present[0, j]) or D.dp[j] > 1.0:
                    continue
                obs.append(rate20(D.Y[r - 1, j, K], t[K]) / rate20(D.Y[0, j, K], t[K]))
                prd.append(rate20(Pm[r - 1, j, K], t[K]) / rate20(Pm[0, j, K], t[K]))
            rows.append(dict(material=NEW_RUNS[r], model="TRACE-Former", density_ratio=float(D.cond[r - 1, 6] / 5600.0),
                             obs_ratio=float(np.nanmedian(obs)), pred_ratio=float(np.nanmedian(prd))))
    return rows


def paper_comparison(t5, dev, test, plans, zs, rr):
    def row(summ, m, a=None):
        g = summ[(summ.method == m) & ((summ.alpha - a).abs() < 1e-9 if a is not None else summ.alpha.isna())]
        return g.iloc[0]
    summ = plans[Y_STAR]
    tf, un, srp = row(summ, "TRACE-Former", 0.05), row(summ, "TRACE-Former (uncalibrated)", 0.05), row(summ, "SRP 6.5.2")
    tfc = [row(plans[ys], "TRACE-Former", 0.05) for ys in Y_STARS]
    cov = lambda U: "/".join(f"{100 * U[l]['coverage']:.1f}" for l in ("80", "90", "95"))
    frac = lambda r: f"{int(r.n_verifiable - r.n_unsafe)}/{int(r.n_verifiable)}"
    rel = lambda r: f"{100 * r.reliability:.0f}% ({frac(r)})"
    z = {r["material"]: r["RMSE"] for r in zs}
    rat = {r["material"]: r for r in rr}
    ratios = lambda key: "/".join(f"{rat[m][key]:.2f}" for m in ("SiC", "TiO2", "CeO2"))
    c, d, t = dev["TRACE-Former (conformal)"], dev, test["TRACE-Former (conformal)"]
    return [("Table 5  CV RMSE", "0.0422", f"{t5['cv_RMSE']:.4f}"),
            ("Table 5  CV logRMSE", "0.259", f"{t5['cv_logRMSE']:.3f}"),
            ("Table 5  test RMSE", "0.0246", f"{t5['test_RMSE']:.4f}"),
            ("Table 5  test logRMSE", "0.146", f"{t5['test_logRMSE']:.3f}"),
            ("Table 5  extrapolation logRMSE", "0.130", f"{t5['extrap_test_logRMSE']:.3f}"),
            ("Section 3.1  extrapolation RMSE", "0.0171", f"{t5['extrap_test_RMSE']:.4f}"),
            ("Table 5  inadmissible trajectories (%)", "0.0", f"{t5['viol_any']:.1f}"),
            ("Table 8  dev. coverage 80/90/95 (%)", "83.2/93.8/98.8", cov(c)),
            ("Table 8  dev. interval score 90%", "0.136", f"{c['90']['interval_score']:.3f}"),
            ("Table 8  dev. CRPS", "0.0176", f"{d['TRACE-Former (conformal)']['CRPS']:.4f}"),
            ("Table 8  test coverage 80/90/95 (%)", "96.4/99.9/100.0", cov(t)),
            ("Table 8  test CRPS", "0.0087", f"{t['CRPS']:.4f}"),
            ("Table 9  TRACE-Former 95% reliability", "100% (17/17)", rel(tf)),
            ("Table 9  TRACE-Former 95% median water (L)", "159", f"{tf.water_reachable_median:.0f}"),
            ("Table 9  uncalibrated 95% reliability", "83% (15/18)", rel(un)),
            ("Table 9  uncalibrated 95% median water (L)", "110", f"{un.water_reachable_median:.0f}"),
            ("Table 9  SRP 6.5.2 median water (L)", "382", f"{srp.water_reachable_median:.0f}"),
            ("Table 9  secondary-check failures TF/unc./SRP", "5/11/10",
             f"{int(tf.n_unsafe_ext)}/{int(un.n_unsafe_ext)}/{int(srp.n_unsafe_ext)}"),
            ("Appendix C  TRACE-Former 95% reliability y*=0.2/0.1/0.05", "19/20, 17/17, 12/12",
             ", ".join(frac(r) for r in tfc)),
            ("Appendix C  TRACE-Former 95% secondary-check failures y*=0.2/0.1/0.05", "2, 5, 5",
             ", ".join(str(int(r.n_unsafe_ext)) for r in tfc)),
            ("Zero-shot RMSE all three materials", "0.112", f"{z['all']:.3f}"),
            ("Zero-shot RMSE SiC", "0.147", f"{z['SiC']:.3f}"),
            ("Zero-shot RMSE CeO2", "0.050", f"{z['CeO2']:.3f}"),
            ("Zero-shot RMSE TiO2", "0.115", f"{z['TiO2']:.3f}"),
            ("Section 3.4  rate ratio to ZrO2 SiC/TiO2/CeO2, predicted", "0.67/0.82/1.20", ratios("pred_ratio")),
            ("Section 3.4  rate ratio to ZrO2 SiC/TiO2/CeO2, measured", "0.41/0.60/1.15", ratios("obs_ratio"))]


def read_json(path):
    with open(path) as f:
        return json.load(f)


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def reproduce(args):
    D = load_data(args.data)
    cfg_path = os.path.join(args.models, "config.json")
    if os.path.exists(cfg_path):
        dp_cfg = np.array(read_json(cfg_path)["dp_um"])
        if len(dp_cfg) != len(D.dp) or np.abs(dp_cfg - D.dp).max() > 1e-9:
            sys.exit("The size channels of the data differ from models/config.json")
    print(f"Data: {args.data} ({N_RUNS} runs, {N_CHANNELS} size channels)", flush=True)
    S16, means, train_rmse, n_params = {}, {}, {}, None
    for name in ["dev26"] + CV_PACKS + ["all32"]:
        hp, members = load_group(args.models, name)
        for m in members:
            model, prep = build_member(hp, m)
            n_params = count_params(model)
            Ys, _ = predict_samples(model, prep, D.cond, D.dp, hp["n_z"])
            means.setdefault(m["split"], {})[m["seed"]] = Ys.mean(0).astype(np.float32)
            if name == "all32":
                train_rmse[m["seed"]] = metrics(np.nan_to_num(D.Y), means["all32"][m["seed"]], eval_mask(D, ZR_RUNS))["RMSE"]
            else:
                S16.setdefault(m["split"], []).append(Ys.astype(np.float16))
        print(f"  predicted {plural(len(members), 'member')} of {name}", flush=True)
    hp30, members30 = load_group(args.models, "t30")
    means30 = [predict_samples(*build_member(hp30, m), D.cond, D.dp, hp30["n_z"])[0].mean(0).astype(np.float32)
               for m in members30]
    print(f"  predicted {plural(len(members30), 'member')} of t30", flush=True)
    seeds = sorted(means["test"])
    for sp in CV_PACKS:
        if sorted(means[sp]) != seeds:
            sys.exit(f"Seeds of {sp} differ from those of dev26")
    t5 = table5(D, means, seeds, n_params, means30)
    print("  Table 5 done", flush=True)
    Y = np.nan_to_num(D.Y)
    Qf = {k: q_trace(S16[f"cv{k}"]) for k in range(5)}
    Qt = q_trace(S16["test"])
    P, w, n_runs = calib_points(D)
    R = cvplus_scores(Qf, Y, P)
    calib = dict(R=[R[P[:, 0] == k] for k in range(5)], W=[w[P[:, 0] == k] for k in range(5)], n_runs=n_runs)
    dev = uq_dev(D, Qf)
    test = uq_test(D, Qf, Qt, calib, len(S16["test"]))
    print("  Table 8 done", flush=True)
    dist, dthr, ad_mu, ad_sd, ad_dev = ad_distance(D)
    scores = calib_scores(D, S16)
    plans = {ys: summarize(planning(D, S16, scores, ys)) for ys in Y_STARS}
    print("  Table 9 and Appendix C done", flush=True)
    zs, Pm = zeroshot(D, means["all32"], train_rmse)
    rr = rate_ratios(D, Pm)
    comp = paper_comparison(t5, dev, test, plans, zs, rr)
    os.makedirs(args.out, exist_ok=True)
    fin = lambda v: None if isinstance(v, float) and not np.isfinite(v) else v
    out = dict(table5=t5, table8={"development_out_of_fold": dev, "test": test},
               planning={f"y_star={ys}": [{k: fin(v) for k, v in r.items()} for r in plans[ys].to_dict("records")]
                         for ys in Y_STARS},
               zeroshot=zs, rate_ratios=rr, ad_threshold=dthr, n_inside_ad=int(sum(d <= dthr for d in dist.values())),
               paper_comparison=[dict(metric=a, paper=b, reproduced=c) for a, b, c in comp])
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    pd.DataFrame([t5]).to_csv(os.path.join(args.out, "table5_accuracy.csv"), index=False)
    t8 = []
    for set_name, U in (("development (out of fold)", dev), ("test", test)):
        for method, r in U.items():
            for lev in ("80", "90", "95"):
                t8.append(dict(run_set=set_name, method=method, level=int(lev), **r[lev], CRPS=r["CRPS"]))
    pd.DataFrame(t8).to_csv(os.path.join(args.out, "table8_uncertainty.csv"), index=False)
    for ys in Y_STARS:
        plans[ys].to_csv(os.path.join(args.out, f"table9_planning_y{ys}.csv"), index=False)
    pd.DataFrame(zs).to_csv(os.path.join(args.out, "zeroshot.csv"), index=False)
    pd.DataFrame(rr).to_csv(os.path.join(args.out, "rate_ratios.csv"), index=False)
    pd.DataFrame(comp, columns=["metric", "paper", "reproduced"]).to_csv(os.path.join(args.out, "paper_comparison.csv"),
                                                                       index=False)
    wa = max(len(a) for a, _, _ in comp)
    wb = max(len(b) for _, b, _ in comp)
    print(f"\n{'metric':<{wa}} | {'paper':<{wb}} | reproduced")
    print("-" * (wa + wb + 18))
    for a, b, c in comp:
        print(f"{a:<{wa}} | {b:<{wb}} | {c}")
    n_diff = sum(b != c for _, b, c in comp)
    print(f"\n{len(comp) - n_diff} of {len(comp)} values equal the paper at the printed precision (all of them with the "
          f"original data set and 2 threads). Results: {args.out}")
    if args.save_calibration:
        path = os.path.join(args.out, "calibration.npz")
        save_calibration(path, calib, scores, ad_mu, ad_sd, ad_dev, D, dthr)
        print(f"wrote {path}\nAfter retraining, copy it into the models directory so that --predict uses it: "
              f"cp {path} {os.path.join(args.models, 'calibration.npz')}")


def save_calibration(path, calib, scores, ad_mu, ad_sd, ad_dev, D, dthr):
    arrays = {}
    for k in range(5):
        o = np.argsort(calib["R"][k], kind="stable")
        arrays[f"cv_scores_fold{k}"] = calib["R"][k][o]
        arrays[f"cv_weights_fold{k}"] = calib["W"][k][o]
    idx = np.asarray(DEV_RUNS) - 1
    arrays.update(n_calib_runs=np.array(calib["n_runs"]), plan_alphas=np.array(PLAN_ALPHAS),
                  plan_qhat=np.array([plan_qhat(scores, a) for a in PLAN_ALPHAS]), ad_mu=ad_mu, ad_sd=ad_sd,
                  ad_dev_features=ad_dev, ad_dev_nozzle=np.array([str(x) for x in D.nozzle[idx]]),
                  ad_dev_Q_lpm=D.cond[idx, 1], ad_dev_mist_dm_um=np.where(D.cond[idx, 5] > 0, D.cond[idx, 4], 0.0),
                  ad_threshold=np.array(dthr), ad_feature_names=np.array(AD_FEATURES))
    np.savez(path, **arrays)


def scenario_number(r, col, where, default=None):
    v = str(r.get(col, "")).strip()
    if v == "":
        if default is None:
            sys.exit(f"{where}: {col} is empty")
        return default
    try:
        x = float(v)
    except ValueError:
        sys.exit(f"{where}: {col} must be a number, got {v!r}")
    if not math.isfinite(x):
        sys.exit(f"{where}: {col} must be a finite number, got {v!r}")
    return x


def read_scenarios(path):
    if not os.path.exists(path):
        sys.exit(f"Scenario file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        sys.exit(f"{path} is empty; give a header line and one scenario per row")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        sys.exit(f"Cannot read the scenario file {path}: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    miss = [c for c in SCENARIO_COLS if c not in df.columns]
    if miss:
        sys.exit(f"Missing columns in {path}: {miss}")
    if df.empty:
        sys.exit(f"{path} contains a header line but no scenario rows")
    rows, seen = [], set()
    for n, r in enumerate(df.to_dict("records"), start=1):
        sid = str(r["id"]).strip()
        where = f"{path}, row {n} (id {sid!r})"
        if not sid or sid in seen:
            sys.exit(f"{where}: every scenario needs a unique, non-empty id")
        seen.add(sid)
        num = lambda col, default=None: scenario_number(r, col, where, default)
        noz = str(r.get("nozzle", "")).strip()
        if noz:
            if noz not in NOZZLES:
                sys.exit(f"{where}: unknown nozzle {noz!r} (use {', '.join(NOZZLES)}, or leave nozzle empty and give "
                         f"Dd_um and orifice_mm)")
            Dd, orf = NOZZLES[noz]
            if num("Dd_um", Dd) != Dd or num("orifice_mm", orf) != orf:
                sys.exit(f"{where}: Dd_um and orifice_mm differ from nozzle {noz} ({Dd:g} um, {orf:g} mm); leave them "
                         f"empty or leave the nozzle empty")
        else:
            if not str(r.get("Dd_um", "")).strip() or not str(r.get("orifice_mm", "")).strip():
                sys.exit(f"{where}: give a nozzle (N1 to N4) or both Dd_um and orifice_mm")
            Dd, orf = num("Dd_um"), num("orifice_mm")
            noz = nozzle_of(Dd, orf)
        vals = [num("coverage"), num("Q_lpm"), Dd, orf, num("mist_dm_um"), num("mist_Cm_cm3"), num("rho_kgm3"),
                num("contact_angle_deg")]
        ys, conf = num("y_star", Y_STAR), num("confidence", 0.95)
        if not 0 < vals[0] <= 1:
            sys.exit(f"{where}: coverage must lie in (0, 1], got {vals[0]:g}")
        if min(vals[1], vals[2], vals[3], vals[6]) <= 0:
            sys.exit(f"{where}: Q_lpm, Dd_um, orifice_mm and rho_kgm3 must be positive")
        if min(vals[4], vals[5]) < 0 or (vals[4] > 0) != (vals[5] > 0):
            sys.exit(f"{where}: give both mist_dm_um and mist_Cm_cm3 as positive values for a mist, or 0 for both "
                     f"(no mist); got {vals[4]:g} and {vals[5]:g}")
        if not 0 <= vals[7] <= 180:
            sys.exit(f"{where}: contact_angle_deg must lie between 0 and 180, got {vals[7]:g}")
        if not min(Y_STARS) <= ys <= max(Y_STARS):
            sys.exit(f"{where}: y_star must lie between {min(Y_STARS):g} and {max(Y_STARS):g} (the removal targets "
                     f"validated in the paper), got {ys:g}")
        if min(abs(conf - c) for c in CONF_LEVELS) > 1e-9:
            sys.exit(f"{where}: confidence must be one of {', '.join(f'{c:g}' for c in CONF_LEVELS)} (the calibrated "
                     f"levels), got {conf:g}")
        rows.append(dict(id=sid, nozzle=noz, cond=vals, y_star=ys, confidence=conf))
    return rows


def predict(args):
    cfg_path = os.path.join(args.models, "config.json")
    cal_path = os.path.join(args.models, "calibration.npz")
    for p in (cfg_path, cal_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}")
    sc = read_scenarios(args.predict)
    dp = np.array(read_json(cfg_path)["dp_um"])
    cal = np.load(cal_path)
    ens = {}
    for name in ["dev26"] + CV_PACKS:
        hp, members = load_group(args.models, name)
        ens[name] = [build_member(hp, m) + (hp["n_z"],) for m in members]
    th = T_PRED / 60.0
    pos = T_PRED > 0
    idx = np.nonzero(np.broadcast_to(pos, (len(dp), len(T_PRED))))
    gg = (dp >= GG_LO) & (dp <= GG_HI)
    gg_text = f"{GG_LO:g}-{GG_HI:.1f} um, {int(gg.sum())} channels"
    Rs = [cal[f"cv_scores_fold{k}"] for k in range(5)]
    Ws = [cal[f"cv_weights_fold{k}"] for k in range(5)]
    thr = float(cal["ad_threshold"])
    traj, plans = [], []
    for s in sc:
        cond = np.array([s["cond"]])
        samples = lambda name: [predict_samples(model, prep, cond, dp, n_z, times=th) for model, prep, n_z in ens[name]]
        dev = samples("dev26")
        S = np.concatenate([y for y, _ in dev], 0)[:, 0]
        par = {k: np.mean([p[k] for _, p in dev], 0)[0] for k in dev[0][1]}
        noisy = add_noise(S, seed=1)
        Qk = [np.quantile(add_noise(np.concatenate([y for y, _ in samples(p)], 0)[:, 0], seed=1), TAUS, axis=0)
              for p in CV_PACKS]
        CS = [cvplus_centers(q, idx) for q in Qk]
        L, U = cvplus_bounds([c for c, _ in CS], [h for _, h in CS], Rs, Ws, 0.1, int(cal["n_calib_runs"]))
        L = -np.sort(-np.clip(L.reshape(len(dp), -1), 0, 1), axis=-1)
        U = -np.sort(-np.clip(U.reshape(len(dp), -1), 0, 1), axis=-1)
        mean = S.mean(0)
        med = np.quantile(noisy, 0.5, axis=0)
        med[:, 0] = 1.0
        med = np.minimum.accumulate(med, axis=-1)
        for j, d in enumerate(dp):
            for kk, t in enumerate(T_PRED):
                first = t == 0
                traj.append(dict(id=s["id"], quantity="C/C0", dp_um=d, t_min=t, mean=1.0 if first else float(mean[j, kk]),
                                 median=float(med[j, kk]), lower=1.0 if first else float(L[j, kk - 1]),
                                 upper=1.0 if first else float(U[j, kk - 1]), level=0.9))
        alpha = min(PLAN_ALPHAS, key=lambda a: abs(1 - s["confidence"] - a))
        qhat = float(cal["plan_qhat"][list(cal["plan_alphas"]).index(alpha)])
        msamp = noisy[:, gg].max(1)
        mmean = S[:, gg].max(1).mean(0)
        Ub = plan_bound(T_PRED, msamp, qhat)
        q50 = np.quantile(msamp, 0.5, axis=0)
        q50[0] = 1.0
        q50 = np.minimum.accumulate(q50)
        for kk, t in enumerate(T_PRED):
            traj.append(dict(id=s["id"], quantity="m_GG", dp_um=np.nan, t_min=t, mean=1.0 if t == 0 else float(mmean[kk]),
                             median=float(q50[kk]), lower=np.nan, upper=float(Ub[kk]), level=s["confidence"]))
        ys = s["y_star"]
        ts = first_cross(loglin(T_PRED, Ub), ys)
        tmed = first_cross(loglin(T_PRED, q50), ys)
        cov_, Q = s["cond"][0], s["cond"][1]
        tsrp = srp_time(cov_, Q, ys)
        dd = np.linalg.norm(cal["ad_dev_features"] - (ad_features(cond)[0] - cal["ad_mu"]) / cal["ad_sd"], axis=1)
        jn = int(np.argmin(dd))
        inside = bool(dd[jn] <= thr)
        material_tested = bool(s["cond"][6] == ZRO2[0] and s["cond"][7] == ZRO2[1])
        if not inside:
            status = "no plan: outside the applicability domain"
        elif not material_tested:
            status = "no plan: material outside the planning validation (zero-shot prediction only)"
        else:
            status = "plan" if np.isfinite(ts) else "target not reached within 600 min"
        planned = status == "plan"
        fac = 2 * V_VESSEL / (3 * H_FALL * (Q * 60 / 1000.0))
        ed0 = fac * par["phib"][0] * par["lam_in"]
        edl = fac * (par["mu2"] - par["lam0"])
        fin = lambda v: float(v) if np.isfinite(v) else np.nan
        plans.append(dict(id=s["id"], nozzle=s["nozzle"], Dd_um=s["cond"][2], orifice_mm=s["cond"][3], Q_lpm=Q,
                          coverage=cov_, mist_dm_um=s["cond"][4], mist_Cm_cm3=s["cond"][5], rho_kgm3=s["cond"][6],
                          contact_angle_deg=s["cond"][7], y_star=ys, confidence=s["confidence"], plan_status=status,
                          t_stop_min=ts if planned else np.nan, water_L=Q * ts if planned else np.nan,
                          stop_within_60min=bool(planned and ts <= 60.0), q_hat=qhat, gg_channels=gg_text,
                          t_median_min=fin(tmed), srp_t_stop_min=fin(tsrp), srp_water_L=fin(Q * tsrp),
                          ad_distance=float(dd[jn]), ad_threshold=thr, inside_ad=inside,
                          nearest_dev_nozzle=str(cal["ad_dev_nozzle"][jn]), nearest_dev_Q_lpm=float(cal["ad_dev_Q_lpm"][jn]),
                          nearest_dev_mist_dm_um=float(cal["ad_dev_mist_dm_um"][jn]),
                          nozzle_tested=s["nozzle"] in NOZZLES, material_tested=material_tested,
                          prior_phi=float(par["phi"][0]), prior_k_per_h=float(par["k"][0]),
                          prior_eps=float(par["eps"][0]), prior_ED_initial_gg_per_m=float(np.median(ed0[gg])),
                          prior_ED_late_gg_per_m=float(np.median(edl[gg]))))
        print(f"  {s['id']}: {status}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(traj).to_csv(os.path.join(args.out, "predicted_trajectories.csv"), index=False)
    P = pd.DataFrame(plans)
    P.to_csv(os.path.join(args.out, "spray_plans.csv"), index=False)
    pd.set_option("display.width", 200)
    print(P[["id", "nozzle", "Q_lpm", "y_star", "confidence", "t_stop_min", "water_L", "srp_water_L", "ad_distance",
             "inside_ad", "material_tested"]].round(3).to_string(index=False))
    print(f"wrote {os.path.join(args.out, 'predicted_trajectories.csv')} and {os.path.join(args.out, 'spray_plans.csv')}")


def main():
    ap = argparse.ArgumentParser(description="Reproduce the TRACE-Former results of the paper from the packed models "
                                             "and the data, or predict new spray scenarios without data (--predict).")
    ap.add_argument("--data", default=DEFAULT_DATA, help="data file (.xlsx with sheet Sheet1, or .csv with the same "
                                                         "headers); default data/aerosol_data.xlsx")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="directory with the packed models (default models)")
    ap.add_argument("--out", default=None, help="output directory (default results, or predictions with --predict)")
    ap.add_argument("--predict", default=None, metavar="CONDITIONS_CSV",
                    help="scenario file for inference without data (columns: id, nozzle or Dd_um and orifice_mm, Q_lpm, "
                         "coverage, mist_dm_um, mist_Cm_cm3, rho_kgm3, contact_angle_deg, optional y_star between 0.05 "
                         "and 0.2 (default 0.1) and confidence 0.5, 0.7, 0.8, 0.9, 0.95 or 0.98 (default 0.95))")
    ap.add_argument("--save-calibration", action="store_true",
                    help="also write calibration.npz (CV+ scores, planning quantiles, applicability domain) to --out; "
                         "after retraining, copy it into the models directory for --predict")
    ap.add_argument("--threads", type=int, default=2, help="torch threads (default 2; the paper's predictions were "
                                                          "made with 2 threads, other values change the last bits)")
    args = ap.parse_args()
    if args.threads < 1:
        sys.exit("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    if args.predict:
        args.out = args.out or os.path.join(ROOT, "predictions")
        predict(args)
    else:
        args.out = args.out or os.path.join(ROOT, "results")
        reproduce(args)


if __name__ == "__main__":
    main()
