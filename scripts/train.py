import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "aerosol_data.xlsx")
DEFAULT_OUT = os.path.join(ROOT, "models")

UM = "(\u03bcm)"
COLS = dict(cov="Spray Coverage", Q="Spray Flow Rate (L/min)", Dd="Droplet mean size  " + UM,
            orf="Orfice diameter (mm)", dm="Mist mean size " + UM, Cm="Mist mean concentration (P/cm^3)",
            rho="Aerosol density (kg/m3)", CA="Contact Angle", dp="Aerosol size " + UM)
COND = ["cov", "Q", "Dd", "orf", "dm", "Cm", "rho", "CA"]
TIME_COLS = ["0min", "1min", "3min", "5min", "10min", "15min", "20min", "30min", "40min", "50min", "60min"]
T_MIN = np.array([0, 1, 3, 5, 10, 15, 20, 30, 40, 50, 60], float)
T_H = T_MIN / 60.0
N_RUNS, N_CHANNELS = 35, 48

V_VESSEL = 3.92
H_VESSEL = 2.5
D_VESSEL = 1.5
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
PACKS = dict(dev26=dict(split="test", tmax=None, seeds="0-9"),
             **{p: dict(split=p, tmax=None, seeds="0-9") for p in CV_PACKS},
             all32=dict(split="all32", tmax=None, seeds="0-9"),
             t30=dict(split="test", tmax=30.0, seeds="0-4"))
GROUPS = dict(dev26=["dev26"], cv=CV_PACKS, all32=["all32"], t30=["t30"], **{p: [p] for p in CV_PACKS})

NOZZLES = {"N1": (309.0, 1.20), "N2": (344.0, 1.70), "N3": (302.0, 1.20), "N4": (162.0, 2.91)}
MISTS = {"none": (0.0, 0.0), "M071": (0.71, 170664.0), "M078": (0.78, 270037.0)}
MATERIALS = {"ZrO2": (5600.0, 66.0), "SiC": (3210.0, 140.0), "CeO2": (7220.0, 46.0), "TiO2": (4230.0, 80.0)}

HP = dict(d=64, heads=4, n_cond=2, n_spec=2, dropout=0.1, dz=3, lr=3e-3, wd=1e-2, steps=3000, beta=10.0,
          lam_res=0.05, lam_warp=0.01, clip=1.0, sig_floor=0.02, beta_init=0.5, beta_aI_init=0.5, n_z=64,
          tmax=None)

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

MISSING_DATA = ("The experimental data set of the paper is provided as data/aerosol_data.xlsx in this repository. "
                "Restore it there or pass a data file in the same layout with --data <file> (.xlsx or .csv), "
                "for example the synthetic example data/example/synthetic_aerosol_data.csv.")


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
    return dict(cond=cond, Y=Y, mask=np.isfinite(Y), dp=dp)


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


def fit_prep(cond):
    x = np.column_stack([np.log(cond[:, 2]), np.log(cond[:, 3])])
    q = np.log(cond[:, 1])
    on = (cond[:, 4] > 0) & (cond[:, 5] > 0)
    m = raw_mist(cond[on]) if on.any() else np.zeros((1, 2))
    return dict(mu_s=x.mean(0), sd_s=np.maximum(x.std(0), 1e-6), mu_q=float(q.mean()), sd_q=float(max(q.std(), 1e-6)),
                mu_c=float(np.log(cond[:, 0]).mean()), mu_m=m.mean(0), sd_m=np.maximum(m.std(0), 1e-6))


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

    def sigma(self, y):
        return torch.sqrt((SIG_REL * y) ** 2 + torch.exp(self.log_sig_floor) ** 2)

    def parts(self, B, z):
        pooled, h = self.enc(B)
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
        return dict(phi=phi, k=k, eps=eps, lam0=lam0, lam_in=lam_in, res=rho0)

    def forward(self, B, z):
        t = B["t"]
        P = self.parts(B, z)
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
        P.update(warp_su=w)
        return y, P


def fit_member(data, name, split, seed, hp, threads):
    torch.set_num_threads(threads)
    t0 = time.time()
    train_runs = SPLITS[split]
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.asarray(train_runs) - 1
    cond = data["cond"][idx]
    prep = fit_prep(cond)
    B = make_batch(cond, data["dp"], prep, T_H)
    Yt = torch.tensor(np.nan_to_num(data["Y"][idx], nan=0.0), dtype=torch.float32)
    M = data["mask"][idx].copy()
    M[:, :, 0] = False
    M = torch.tensor(M)
    if hp["tmax"] is not None:
        M = M & (B["t"] <= hp["tmax"] / 60.0 + 1e-9).view(1, 1, -1)
    model = TraceFormer(hp)
    with torch.no_grad():
        model.log_sig_floor.fill_(math.log(hp["sig_floor"]))
    model.log_sig_floor.requires_grad_(False)
    R, dz = len(train_runs), model.dz
    q_mu = torch.zeros(R, dz, requires_grad=True)
    q_ls = torch.full((R, dz), math.log(0.5), requires_grad=True)
    params = [p for p in model.parameters() if p.requires_grad] + [q_mu, q_ls]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=hp["steps"], pct_start=0.1)
    n_pts = M.sum()
    for step in range(hp["steps"]):
        model.train()
        z = (q_mu + torch.exp(q_ls) * torch.randn_like(q_mu))[:, :dz]
        y, P = model(B, z)
        sig = model.sigma(y.detach())
        nll = (0.5 * ((Yt - y) / sig) ** 2 + torch.log(sig))[M].sum() / n_pts
        loss = nll
        kl = 0.5 * (q_mu ** 2 + torch.exp(2 * q_ls) - 1 - 2 * q_ls).sum() / n_pts
        loss = loss + hp["beta"] * kl
        loss = loss + hp["lam_res"] * (P["res"] ** 2).mean()
        loss = loss + hp["lam_warp"] * P["warp_su"] ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, hp["clip"])
        opt.step()
        sched.step()
    model.eval()
    print(f"  {name} seed {seed}: {hp['steps']} steps, final NLL {float(nll.detach()):.4f}, {time.time() - t0:.0f} s", flush=True)
    prep_t = {k: (torch.tensor(v, dtype=torch.float64) if isinstance(v, np.ndarray) else float(v)) for k, v in prep.items()}
    return dict(split=split, seed=int(seed), train_runs=[int(r) for r in train_runs],
                state={k: v.detach().clone() for k, v in model.state_dict().items()}, prep=prep_t)


def pack(name, hp, members):
    return {"format": "trace-former-v1", "name": name, "hp": dict(hp), "members": members}


def config_dict(hp, dp):
    return {"format": "trace-former-v1", "hp": dict(hp), "dp_um": [float(v) for v in dp],
            "time_columns": TIME_COLS, "t_min": [float(v) for v in T_MIN],
            "t_min_zro2": [float(v) for v in T_MIN if v != 15], "t_min_new_materials": [float(v) for v in T_MIN if v <= 20],
            "splits": {"FOLD_RUNS": FOLD_RUNS, "TEST_RUNS": TEST_RUNS, "DEV_RUNS": DEV_RUNS, "ZR_RUNS": ZR_RUNS,
                       "NEW_RUNS": {str(k): v for k, v in NEW_RUNS.items()}},
            "model_files": {p: {"file": f"trace_former_{p}.pt", "split": s["split"], "train_runs": SPLITS[s["split"]],
                                "training_window_min": [0.0, s["tmax"] or 60.0], "paper_seeds": s["seeds"]}
                            for p, s in PACKS.items()},
            "nozzles": {k: {"Dd_um": v[0], "orifice_mm": v[1]} for k, v in NOZZLES.items()},
            "mists": {k: {"mist_dm_um": v[0], "mist_Cm_cm3": v[1]} for k, v in MISTS.items()},
            "materials": {k: {"rho_kgm3": v[0], "contact_angle_deg": v[1]} for k, v in MATERIALS.items()},
            "vessel": {"volume_m3": V_VESSEL, "height_m": H_VESSEL, "diameter_m": D_VESSEL, "fall_height_m": H_FALL},
            "srp_6_5_2": {"E_over_D_high_per_m": SRP_ED_HIGH, "E_over_D_low_per_m": SRP_ED_LOW,
                          "DF_switch": SRP_DF_SWITCH, "air_changes_per_h": SRP_ACH,
                          "coverage_single_region": SRP_COV_UNIFORM}}


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def parse_seeds(text):
    seeds = []
    try:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                seeds += list(range(int(a), int(b) + 1))
            elif part:
                seeds.append(int(part))
    except ValueError:
        sys.exit(f"Invalid --seeds {text!r}; use e.g. 0-9 or 0,1,2")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        sys.exit(f"Invalid --seeds {text!r}; give distinct non-negative seeds, e.g. 0-9 or 0,1,2")
    return seeds


def main():
    ap = argparse.ArgumentParser(description="Train the TRACE-Former ensembles (paper protocol: seeds 0-9 on the 26 "
                                             "development runs, on each of the 5 cross-validation folds and on all "
                                             "32 ZrO2 runs, and seeds 0-4 on the 0-30 min data of the 26 development "
                                             "runs).")
    ap.add_argument("--data", default=DEFAULT_DATA, help="data file (.xlsx with sheet Sheet1, or .csv with the same "
                                                         "headers); default data/aerosol_data.xlsx")
    ap.add_argument("--groups", default="dev26,cv,all32,t30",
                    help="comma list of dev26, cv (the five folds; single folds cv0 to cv4 are also accepted), all32 "
                         "and t30 (default: dev26,cv,all32,t30)")
    ap.add_argument("--seeds", default=None, help="seeds for every selected group, e.g. 0-9 or 0,1,2 (default: 0-9, "
                                                  "and 0-4 for t30)")
    ap.add_argument("--steps", type=int, default=HP["steps"], help="full-batch optimisation steps (default 3000)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="parallel processes (default min(8, CPU count))")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process (default 2, as in the paper; "
                                                          "bitwise reproduction of the shipped members needs 2)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default models)")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad or not groups:
        sys.exit(f"Unknown or missing group(s) {bad}; choose from dev26, cv, cv0 to cv4, all32, t30")
    if args.steps < 1 or args.workers < 1 or args.threads < 1:
        sys.exit("--steps, --workers and --threads must be at least 1")
    names = [p for p in PACKS if any(p in GROUPS[g] for g in groups)]
    seeds = {p: parse_seeds(args.seeds or PACKS[p]["seeds"]) for p in names}
    data = load_data(args.data)
    hps = {p: dict(HP, steps=args.steps, tmax=PACKS[p]["tmax"]) for p in names}
    jobs = [(data, p, PACKS[p]["split"], seed, hps[p], args.threads) for p in names for seed in seeds[p]]
    os.makedirs(args.out, exist_ok=True)
    workers = min(args.workers, len(jobs))
    print(f"Training {plural(len(jobs), 'member')} ({', '.join(f'{p}: seeds {seeds[p]}' for p in names)}) with "
          f"{plural(workers, 'worker')} x {plural(args.threads, 'thread')}", flush=True)
    t0 = time.time()
    if workers == 1:
        results = [fit_member(*j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(fit_member, jobs)
    for p in names:
        members = [m for j, m in zip(jobs, results) if j[1] == p]
        path = os.path.join(args.out, f"trace_former_{p}.pt")
        torch.save(pack(p, hps[p], members), path)
        print(f"wrote {path} ({plural(len(members), 'member')})")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config_dict(dict(HP, steps=args.steps), data["dp"]), f, indent=1)
    print(f"wrote {os.path.join(args.out, 'config.json')}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
