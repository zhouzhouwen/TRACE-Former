# Data

## Data set

`data/aerosol_data.xlsx` is the final experimental data set of the paper: 35 size-resolved runs of the UTARTS facility, 32 with ZrO₂ aerosols and one each with SiC, CeO₂, and TiO₂. Please cite the paper when you use the data.

## File format

`scripts/train.py` and `scripts/test.py` read `data/aerosol_data.xlsx` (sheet `Sheet1`) by default. Another file in the same layout, also a CSV file with the same headers, can be passed with `--data`. Each row holds the trajectory of one aerosol-size channel in one run. Other columns, such as the run label `Case` and the empty column `Unnamed: 20` of the experimental workbook, are ignored.

## Column dictionary

| Column header | Unit | Meaning |
|:--|:--|:--|
| `Spray Coverage` | – | nominal spray coverage, the sprayed fraction of the vessel volume |
| `Spray Flow Rate (L/min)` | L/min | spray flow rate |
| `Droplet mean size  (μm)` | μm | droplet mean size of the nozzle |
| `Orfice diameter (mm)` | mm | orifice diameter of the nozzle |
| `Mist mean size (μm)` | μm | mist mean size, 0 without mist |
| `Mist mean concentration (P/cm^3)` | cm⁻³ | mist number concentration, 0 without mist |
| `Aerosol density (kg/m3)` | kg/m³ | density of the aerosol material |
| `Contact Angle` | ° | water contact angle of the aerosol material |
| `Aerosol size (μm)` | μm | aerosol-size channel, one of 48 channels between 0.198 and 5.833 μm |
| `0min`, `1min`, `3min`, `5min`, `10min`, `15min`, `20min`, `30min`, `40min`, `50min`, `60min` | – | normalized concentration C/C0 at the sampling time |

The headers must match exactly, including the two spaces between `size` and `(μm)` in `Droplet mean size  (μm)`, the spelling `Orfice`, and the Greek letter μ (U+03BC). The operating conditions are repeated in every row of a run. C/C0 is 1 at 0 min, 0 where the concentration fell below the detection limit, and empty at times without a sample. The ZrO₂ runs were sampled at 0, 1, 3, 5, 10, 20, 30, 40, 50, and 60 min and the other materials at 0, 1, 3, 5, 10, 15, and 20 min. A channel without a measurement in a run has no row. The experimental workbook has 1,500 rows, 1,365 for the ZrO₂ runs and 135 for the other materials.

## Run structure

The rows list the 35 runs in order, with the channels of each run in ascending aerosol size. The scripts start a new run wherever the aerosol size does not increase from one row to the next and stop with an error unless they find 35 runs and 48 distinct channels. Runs 1 to 32 used ZrO₂ aerosols (5600 kg/m³, 66°) and were measured for 60 min. Runs 33 (SiC), 34 (CeO₂), and 35 (TiO₂) were measured for 20 min under the conditions of run 1. The run numbers define the splits:

| Set | Runs | Use |
|:--|:--|:--|
| Test runs | 4, 5, 15, 16, 29, 30 | held-out evaluation |
| Development runs | the other 26 of runs 1 to 32 | training of `dev26` and `t30`, cross-validation, and calibration |
| Fold 0 | 1, 2, 3, 13, 14 | held out by `cv0` |
| Fold 1 | 8, 9, 10, 22, 23 | held out by `cv1` |
| Fold 2 | 17, 18, 19, 27, 28 | held out by `cv2` |
| Fold 3 | 20, 21, 24, 25, 26 | held out by `cv3` |
| Fold 4 | 6, 7, 11, 12, 31, 32 | held out by `cv4` |
| All ZrO₂ runs | 1 to 32 | training of `all32` |
| New materials | 33, 34, 35 | zero-shot evaluation |

Each cross-validation ensemble `cvk` is trained on the development runs outside fold k. `t30` uses only the samples up to 30 min. `models/config.json` lists the training runs of every model file.

## Example files

`data/example/synthetic_aerosol_data.csv` contains model-generated values, not measurements, for testing the pipeline. It has the headers, run order, and operating conditions of the experimental data set with all 48 channels in every run (1,680 rows). The trajectories of each run come from one randomly chosen `all32` member with one latent draw and measurement noise, clipped to [0, 1] and rounded to four decimals. The `Case` column labels each run as "synthetic run N". Results computed from this file do not reproduce the paper.

`data/example/example_conditions.csv` defines eight scenarios for `python scripts/test.py --predict data/example/example_conditions.csv`, one per row:

| Column | Meaning |
|:--|:--|
| `id` | unique scenario name |
| `nozzle` | `N1`, `N2`, `N3`, or `N4`, or empty when `Dd_um` and `orifice_mm` are given |
| `Dd_um`, `orifice_mm` | droplet mean size (μm) and orifice diameter (mm), empty when `nozzle` is given |
| `Q_lpm` | spray flow rate (L/min) |
| `coverage` | nominal spray coverage, between 0 and 1 |
| `mist_dm_um`, `mist_Cm_cm3` | mist mean size (μm) and number concentration (cm⁻³), both 0 without mist |
| `rho_kgm3`, `contact_angle_deg` | aerosol density (kg/m³) and water contact angle (°) |
| `y_star` | removal target for the slowest Greenfield-gap channel, between 0.05 and 0.2, empty for 0.1 |
| `confidence` | confidence level of the plan, 0.5, 0.7, 0.8, 0.9, 0.95, or 0.98, empty for 0.95 |

The scenarios cover the four nozzles with and without mist, a target of 0.05, confidence levels of 0.9 and 0.98, nozzle N3 given by its droplet size and orifice, and one SiC scenario, which receives a prediction without a plan. The experiments used the nozzles N1 (309 μm, 1.20 mm), N2 (344 μm, 1.70 mm), N3 (302 μm, 1.20 mm), and N4 (162 μm, 2.91 mm), flow rates of 2.0 to 4.0 L/min, nominal coverages of 0.16 to 0.23 (N1), 0.47 to 0.52 (N2 and N3), and 0.87 to 0.96 (N4), and the mists of 0.71 μm at 170,664 cm⁻³ and 0.78 μm at 270,037 cm⁻³. Scenarios far from these settings fall outside the applicability domain and receive no plan.
