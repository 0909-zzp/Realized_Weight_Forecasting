"""Yearly summary for walk-forward weights (read-only on original data)."""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "图形Lasso" / "code"))
from 共享模块 import K, ETA

def load_series():
    valid = np.load(PROJ / "特征工程" / "valid_indices.npy")
    n = len(np.load(PROJ / "特征工程" / "X_features.npy"))
    n_train = int(n*0.70); n_val = int(n*0.15); test_start = n_train + n_val
    npy_dir = PROJ / "数据" / "1min_log_return_npy"
    all_files = sorted([f for f in npy_dir.iterdir() if f.suffix==".npy" and f.name[0].isdigit()])
    idx = valid[test_start:]
    files = [all_files[i] for i in idx]
    simple = np.array([np.expm1(np.load(f).sum(axis=1)) for f in files])
    dates = [f.name[:8] for f in files]
    return files, simple, dates

def net(Y, simple):
    T = len(Y)-1
    pr = np.array([Y[t] @ simple[t+1] for t in range(T)])
    to = np.zeros(T)
    for t in range(T):
        wp = Y[t]; rn = simple[t+1]
        wd = wp*(1+rn)/(1+wp@rn)
        to[t] = np.abs(Y[t+1]-wd).sum()
    return pr - ETA*to, to

files, simple, dates = load_series()
W5 = np.load(OUT/"W_base_walkforward.npy")
W6 = np.load(OUT/"W_dfl_walkforward.npy")
Yeq = np.ones((len(files), K))/K

out_rows = []
for name, Y in [("M5 walk-forward", W5), ("M5+DFL walk-forward", W6), ("Equal Weight", Yeq)]:
    nr, to = net(Y, simple)
    ret_dates = pd.to_datetime(dates[1:], format="%Y%m%d")
    df = pd.DataFrame({"ret": nr, "to": to, "year": ret_dates.year})
    for y, g in df.groupby("year"):
        sd = g["ret"].std(ddof=1)
        out_rows.append({
            "model": name, "year": int(y), "days": len(g),
            "avg_daily": g["ret"].mean(),
            "sharpe": g["ret"].mean()/sd*np.sqrt(252) if sd>0 else 0,
            "avg_turnover": g["to"].mean(),
            "cum": np.prod(1+g["ret"])-1,
        })
res = pd.DataFrame(out_rows).pivot(index="year", columns="model", values=["sharpe","avg_turnover","cum","days"])
flat = pd.DataFrame(out_rows).sort_values(["model","year"])
flat.to_csv(OUT/"walkforward_yearly.csv", index=False)
print(flat.pivot(index="year", columns="model", values="sharpe").round(3).to_string())
print("\navg turnover by year:")
print(flat.pivot(index="year", columns="model", values="avg_turnover").round(4).to_string())
print("\ncum by year:")
print(flat.pivot(index="year", columns="model", values="cum").round(4).to_string())
