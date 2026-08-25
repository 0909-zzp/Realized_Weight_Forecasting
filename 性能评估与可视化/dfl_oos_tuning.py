"""Proper temporal split for DFL parameters, including risk tuning.

Protocol (train / validation / test, no overlap):
  train      : X rows [0:1690]
  validation : 2017-04-11 ~ 2018-09-20 (362 days)
  test       : 2018-09-21 ~ 2020-04-03 (363 days)
  holdout    : test[60:260] = 2018-12-19 ~ 2019-10-08 (Table3 window)

The DFL objective is
  min 0.5*risk_mult*w'Sigma*w + eta*||w - w_drifted||_1
      + 0.5*rho*||w - w_stat||^2

All parameters (eta, rho, risk_mult) are selected on the validation window.
Three selection criteria are reported: max Sharpe, min max drawdown (with a
validation Sharpe floor), and max Calmar. The chosen configurations are then
evaluated once on the holdout.
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "图形Lasso" / "code"))
from 共享模块 import K, ETA, log, set_log_file, load_day, compute_raw_cov, EPS_RIDGE


DFL_BOX = 0.05
RHO_ADMM = 0.01
ROLL_WINDOW = 40
EVAL_START, EVAL_STOP = 60, 260
OUT_CSV = Path(__file__).with_name("dfl_oos_tuning.csv")
OUT_CSV_FINAL = Path(__file__).with_name("dfl_oos_holdout.csv")


def load_parts():
    feat_dir = ROOT / "特征工程"
    varx_dir = ROOT / "VARX"
    valid_indices = np.load(feat_dir / "valid_indices.npy")
    X = np.load(feat_dir / "X_features.npy")
    Y_targets = np.load(feat_dir / "Y_targets.npy")

    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    X_val = X[n_train : n_train + n_val]
    Y_val = Y_targets[n_train : n_train + n_val]
    val_indices = valid_indices[n_train : n_train + n_val]
    test_indices = valid_indices[n_train + n_val :]
    train_day_indices = valid_indices[:n_train]

    model_dir = varx_dir / "fitted_models"
    coefs = np.load(model_dir / "coefs_model4.npy")
    intercepts = np.load(model_dir / "intercepts_model4.npy")
    cols = np.load(model_dir / "feat_cols_model4.npy")
    with open(model_dir / "scaler_model4.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_sel = X_val[:, cols]
    X_std = scaler.transform(X_sel)
    Y_pred_val = X_std @ coefs.T + intercepts
    Y_pred_val = Y_pred_val / Y_pred_val.sum(1, keepdims=True)

    Y_pred_test = np.load(varx_dir / "Y_pred_model4.npy")
    Y_test = Y_targets[n_train + n_val :]

    data_dir = next(d for d in ROOT.rglob("1min_log_return_npy") if d.is_dir())
    all_files = sorted(
        [f for f in data_dir.iterdir() if f.suffix == ".npy" and f.name[0].isdigit()]
    )
    simple_rets_val = np.array(
        [np.expm1(np.load(all_files[i]).sum(axis=1)) for i in val_indices]
    )
    simple_rets_test = np.array(
        [np.expm1(np.load(all_files[i]).sum(axis=1)) for i in test_indices[:260]]
    )
    return (
        Y_pred_val,
        Y_val,
        Y_pred_test,
        Y_test,
        train_day_indices,
        simple_rets_val,
        simple_rets_test,
        all_files,
        test_indices,
    )


def rolling_sigma(train_day_indices, win=40):
    cov_sum = np.zeros((K, K))
    loaded = 0
    for idx in train_day_indices[-win:]:
        try:
            rett = load_day(idx)
            cov = compute_raw_cov(rett)
            cov.flat[:: K + 1] += EPS_RIDGE
            cov_sum += cov
            loaded += 1
        except Exception:
            pass
    return cov_sum / loaded


def _solve_l2_day(w_stat, w_prev, Sigma, eta_vec, rho, risk_mult, box=DFL_BOX):
    A = risk_mult * Sigma + np.diag(eta_vec + rho)
    b = eta_vec * w_prev + rho * w_stat
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.inv(A + 1e-8 * np.eye(K))
    ones = np.ones(K)
    A_inv_b = A_inv @ b
    A_inv_1 = A_inv @ ones
    lam = (A_inv_b.sum() - 1.0) / A_inv_1.sum()
    w = np.clip(A_inv_b - lam * A_inv_1, -box, box)
    return w / w.sum()


def dfl_drift_l1(Y_pred, simple_rets, w0, Sigma, eta, rho, risk_mult=1.0,
                 box=DFL_BOX):
    Q = risk_mult * Sigma + rho * np.eye(K)
    Minv = np.linalg.inv(Q + RHO_ADMM * np.eye(K))
    Minv1 = Minv @ np.ones(K)
    thr = eta / RHO_ADMM
    T = len(Y_pred)
    Y_dfl = np.zeros_like(Y_pred)

    def drift(w, r):
        return w * (1 + r) / (1 + w @ r)

    def solve_l1(w_stat, w_prev):
        c = -rho * w_stat
        z = w_stat.copy()
        u = np.zeros(K)
        for _ in range(1000):
            z_old = z.copy()
            d = c - RHO_ADMM * (z - u)
            w_hat = -Minv @ d
            lam = (w_hat.sum() - 1.0) / Minv1.sum()
            w = w_hat - lam * Minv1
            a = w + u
            z_new = w_prev + np.sign(a - w_prev) * np.maximum(
                np.abs(a - w_prev) - thr, 0.0
            )
            z_new = np.clip(z_new, -box, box)
            u = u + w - z_new
            if (np.linalg.norm(w - z_new) < 1e-9 * np.linalg.norm(w) + 1e-10
                    and RHO_ADMM * np.linalg.norm(z_new - z_old)
                    < 1e-9 * np.linalg.norm(u) + 1e-10):
                z = z_new
                break
            z = z_new
        return z

    for t in range(T):
        w_prev = w0 if t == 0 else drift(Y_dfl[t - 1], simple_rets[t])
        if eta == 0.0:
            Y_dfl[t] = _solve_l2_day(
                Y_pred[t], w_prev, Sigma, np.zeros(K), rho, risk_mult, box
            )
        else:
            Y_dfl[t] = solve_l1(Y_pred[t], w_prev)
    return Y_dfl


def precompute_minvs(sigmas, rho, risk_mult=1.0):
    """Precompute (Minv, Minv1) for each day's rolling covariance."""
    out = []
    eye = np.eye(K)
    for S in sigmas:
        Q = risk_mult * S + rho * eye
        Minv = np.linalg.inv(Q + RHO_ADMM * eye)
        out.append((Minv, Minv @ np.ones(K)))
    return out


def dfl_drift_l1_rolling(Y_pred, simple_rets, w0, sigmas, minvs,
                         eta, rho, risk_mult=1.0, box=DFL_BOX):
    """Drift-aware L1 DFL with a per-day rolling covariance matrix."""
    T = len(Y_pred)
    Y_dfl = np.zeros_like(Y_pred)
    thr = eta / RHO_ADMM

    def drift(w, r):
        return w * (1 + r) / (1 + w @ r)

    def solve_l1(w_stat, w_prev, Minv, Minv1):
        c = -rho * w_stat
        z = w_stat.copy()
        u = np.zeros(K)
        for _ in range(1000):
            z_old = z.copy()
            d = c - RHO_ADMM * (z - u)
            w_hat = -Minv @ d
            lam = (w_hat.sum() - 1.0) / Minv1.sum()
            w = w_hat - lam * Minv1
            a = w + u
            z_new = w_prev + np.sign(a - w_prev) * np.maximum(
                np.abs(a - w_prev) - thr, 0.0
            )
            z_new = np.clip(z_new, -box, box)
            u = u + w - z_new
            if (np.linalg.norm(w - z_new) < 1e-9 * np.linalg.norm(w) + 1e-10
                    and RHO_ADMM * np.linalg.norm(z_new - z_old)
                    < 1e-9 * np.linalg.norm(u) + 1e-10):
                z = z_new
                break
            z = z_new
        return z

    for t in range(T):
        w_prev = w0 if t == 0 else drift(Y_dfl[t - 1], simple_rets[t])
        if eta == 0.0:
            Y_dfl[t] = _solve_l2_day(
                Y_pred[t], w_prev, sigmas[t], np.zeros(K), rho,
                risk_mult, box,
            )
        else:
            Minv, Minv1 = minvs[t]
            Y_dfl[t] = solve_l1(Y_pred[t], w_prev, Minv, Minv1)
    return Y_dfl


def portfolio_metrics(Y, simple_rets, covs=None):
    T = len(Y) - 1
    ret = np.array([Y[t] @ simple_rets[t + 1] for t in range(T)])
    to_vec = np.zeros(T)
    rpv_vec = np.zeros(T)
    for t in range(T):
        w_prev = Y[t]
        r_next = simple_rets[t + 1]
        w_drifted = w_prev * (1 + r_next) / (1 + w_prev @ r_next)
        to_vec[t] = np.sum(np.abs(Y[t + 1] - w_drifted))
        if covs is not None:
            rpv_vec[t] = Y[t] @ covs[t + 1] @ Y[t]
    net = ret - ETA * to_vec
    vol = float(np.std(net, ddof=1)) * np.sqrt(252)
    sr = (
        float(np.mean(net) / np.std(net, ddof=1)) * np.sqrt(252)
        if np.std(net) > 1e-15
        else 0.0
    )
    cum = np.cumprod(1 + net)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak))
    out = {
        "vol_annual": vol,
        "turnover": float(np.mean(to_vec)),
        "sharpe_net": sr,
        "max_dd": dd,
        "cum_ret": float(np.prod(1 + net) - 1),
        "calmar": sr / max(0.01, abs(dd)),
    }
    if covs is not None:
        out["rpv_annual"] = float(np.mean(rpv_vec)) * 252
    return out


def main():
    set_log_file(Path(__file__).with_name("dfl_oos_tuning_log.txt"))
    log("=" * 70)
    log("DFL OOS tuning with risk_mult: validation selection, holdout eval")
    log("=" * 70)

    (
        Y_pred_val,
        Y_val,
        Y_pred_test,
        Y_test,
        train_day_indices,
        simple_rets_val,
        simple_rets_test,
        all_files,
        test_indices,
    ) = load_parts()
    Sigma = rolling_sigma(train_day_indices)
    log(f"validation: {len(Y_pred_val)} days, holdout: test[{EVAL_START}:{EVAL_STOP}]")

    eta_grid = [1e-7, 1e-6, 1e-5]
    rho_grid = [1e-4, 1e-3, 1e-2]
    risk_grid = [1.0, 2.0, 5.0, 10.0]
    rows = []
    t0 = time.time()
    for eta in eta_grid:
        for rho in rho_grid:
            for risk_mult in risk_grid:
                Y_dfl_val = dfl_drift_l1(
                    Y_pred_val, simple_rets_val, Y_val[0], Sigma,
                    eta, rho, risk_mult,
                )
                m = portfolio_metrics(Y_dfl_val, simple_rets_val)
                m.update(
                    {"eta": eta, "rho": rho, "risk_mult": risk_mult}
                )
                rows.append(m)
                log(
                    f"eta={eta:.0e} rho={rho:.0e} risk={risk_mult:.0f}  "
                    f"sharpe={m['sharpe_net']:+.3f} to={m['turnover']:.3f} "
                    f"maxdd={m['max_dd']:.3f} calmar={m['calmar']:.2f}"
                )
    log(f"validation grid done ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    usable = df[(df["turnover"] >= 0.01) & (df["turnover"] <= 0.50)].copy()

    def pick_rule(rule):
        if usable.empty:
            return None
        if rule == "sharpe":
            return usable.loc[usable["sharpe_net"].idxmax()]
        if rule == "maxdd":
            sub = usable[usable["sharpe_net"] >= 0.70]
            if sub.empty:
                return None
            return sub.loc[sub["max_dd"].idxmax()]
        if rule == "calmar":
            return usable.loc[usable["calmar"].idxmax()]
        raise ValueError(rule)

    candidates = []
    for rule, label in [("sharpe", "best_sharpe"),
                        ("maxdd", "min_maxdd"),
                        ("calmar", "best_calmar")]:
        row = pick_rule(rule)
        if row is None:
            continue
        candidates.append(
            (label, float(row["eta"]), float(row["rho"]), float(row["risk_mult"]))
        )
        log(
            f"\nValidation pick {label}: eta={row['eta']:.0e} rho={row['rho']:.0e} "
            f"risk={row['risk_mult']:.0f} sharpe={row['sharpe_net']:+.3f} "
            f"to={row['turnover']:.3f} maxdd={row['max_dd']:.3f}"
        )

    covs_test = []
    for i in test_indices[:260]:
        rett = np.load(str(all_files[i]))
        raw = compute_raw_cov(rett)
        raw.flat[:: K + 1] += EPS_RIDGE
        covs_test.append(raw)

    final_rows = []
    for label, eta, rho, risk_mult in candidates:
        Y_dfl_test = dfl_drift_l1(
            Y_pred_test[:260], simple_rets_test, Y_test[0], Sigma,
            eta, rho, risk_mult,
        )
        Y_eval = Y_dfl_test[EVAL_START:EVAL_STOP]
        m = portfolio_metrics(
            Y_eval,
            simple_rets_test[EVAL_START:EVAL_STOP],
            covs_test[EVAL_START:EVAL_STOP],
        )
        m.update(
            {"label": label, "eta": eta, "rho": rho, "risk_mult": risk_mult}
        )
        final_rows.append(m)
        log(
            f"\nholdout {label}: eta={eta:.0e} rho={rho:.0e} risk={risk_mult:.0f} "
            f"vol={m['vol_annual']:.4f} rpv={m.get('rpv_annual', np.nan):.6f} "
            f"to={m['turnover']:.3f} sharpe={m['sharpe_net']:+.3f} "
            f"maxdd={m['max_dd']:.3f} cum={m['cum_ret']:+.3f}"
        )

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT_CSV_FINAL, index=False)
    log(f"\nSaved: {OUT_CSV}")
    log(f"Saved: {OUT_CSV_FINAL}")


if __name__ == "__main__":
    main()
