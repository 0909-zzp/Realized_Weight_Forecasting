"""组合净收益损失版 MCS —— 统一到 Table 4 的窗口与推断设定。

与权重损失版 (run_table4_mcs_blocklen.py) 的唯一差别是损失定义。窗口(测试块
[65:265], 200 天)、块长网格(5/10/20)、bootstrap 次数(2000)、种子(42)、
NW-HAC 滞后阶(floor(4(T/100)^(2/9))) 全部相同, 两套 p 值可直接并排比较。

成交滞后有两种口径, 本脚本两种都算并报告:
  same: Y[t] 赚 t 日收益 —— 与 特征工程 的目标定义一致(第 t 行目标权重由 t 日
        协方差构造, 特征只用 t-1 及更早), 故预测在 t-1 收盘即可下单;
  next: Y[t] 赚 t+1 日收益 —— Table 3_投资组合表现.py 采用的滞后一天成交口径。
被替代的旧脚本 MCS_投资组合.py 用的是早 5 天的窗口(测试块 [60:260])和 iid 标准误。

输出:
  Table4_MCS_portfolio.csv          每模型 × 每口径 × 每块长 的 p 值与成员资格
  Table4_MCS_portfolio_summary.csv  净收益/夏普/换手/两套 MCS p 值
  Table4_MCS_portfolio_LaTeX.tex    权重损失 vs 组合损失 并排表
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = PROJ / "性能评估与可视化"
VARX = PROJ / "VARX"
sys.path.insert(0, str(PROJ / "图形Lasso" / "code"))
from 共享模块 import K, ETA, load_day  # noqa: E402

spec = importlib.util.spec_from_file_location("mcs", str(OUT / "MCS_完整程序.py"))
mcs = importlib.util.module_from_spec(spec)
sys.modules["mcs"] = mcs
spec.loader.exec_module(mcs)

names = ["VAR", "Sparse VAR", "Sparse VARX", "Network VARX",
         "Network VARX-S", "Network VARX-S-DF", "LSTM"]
files = ["Y_pred_model1.npy", "Y_pred_model2.npy", "Y_pred_model3.npy",
         "Y_pred_model4.npy", "Y_pred_model5.npy", "Y_pred_model6_opt.npy",
         "Y_pred_model7.npy"]
mcs.MODEL_NAMES = {i + 1: names[i] for i in range(len(names))}

# ---- Table 4 的 200 天窗口: 363 天测试块中的位置 [65:265] ----
valid = np.load(PROJ / "特征工程" / "valid_indices.npy")
n = len(np.load(PROJ / "特征工程" / "X_features.npy"))
test_start = int(n * .7) + int(n * .15)
POS = np.arange(65, 265)
days_ext = np.append(valid[test_start + POS], valid[test_start + POS[-1] + 1])
print(f"窗口: 测试块位置 {POS[0]}..{POS[-1]}  绝对日 {days_ext[0]}..{days_ext[-1]}  "
      f"{len(POS)}天评估  K={K}  ETA={ETA:g}")

ret_ext = np.array([np.expm1(load_day(int(d)).sum(axis=1)) for d in days_ext])
Yall = {nm: np.load(VARX / f) for nm, f in zip(names, files)}
print("预测数组:", next(iter(Yall.values())).shape)

net_by, tno_by = {}, {}
for align, sl in [("same", (0, 200)), ("next", (1, 201))]:
    rets = ret_ext[slice(*sl)]
    net, tno = {}, {}
    for nm in names:
        Y = Yall[nm]
        w, w_next = Y[POS], Y[POS + 1]
        gross = (w * rets).sum(axis=1)
        drifted = w * (1.0 + rets) / (1.0 + gross)[:, None]
        to = np.abs(w_next - drifted).sum(axis=1)
        net[nm] = gross - ETA * to
        tno[nm] = to
    net_by[align] = pd.DataFrame(net)
    tno_by[align] = pd.DataFrame(tno)

# ---- 与权重损失版同一套 MCS 推断 ----
T = len(POS)
M = len(names)
lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
rows, margin_rows, dm_rows, corr_stat = [], [], [], {}
for align, net in net_by.items():
    L_port = (-net).to_numpy()
    c = pd.DataFrame(L_port).corr().to_numpy()
    o = c[~np.eye(M, dtype=bool)]
    corr_stat[align] = (o.min(), np.median(o), o.max())
    for blk in mcs.BLOCK_LENGTHS:
        rounds, last = mcs.mcs_path(L_port, mcs.N_BOOTSTRAP, blk, mcs.SEED, verbose=False)
        sets = mcs.mcs_member_sets(rounds, M)
        for i, nm in enumerate(names):
            rec = next((r for r in rounds if r["eliminated"] == i), None)
            rows.append({"trade_lag": align, "block_len": blk, "model": nm,
                         "elim_round": -1 if rec is None else rec["round"],
                         "MCS_pval": 1.0 if rec is None else rec["p_hat"],
                         "in_MCS_75": i in sets[0.25][0], "in_MCS_90": i in sets[0.10][0],
                         "set_size_75": len(sets[0.25][0]), "set_size_90": len(sets[0.10][0])})
        margin_rows.append({"trade_lag": align, "block_len": blk,
                            "max_p_hat": max(r["p_hat"] for r in rounds),
                            "min_T_R": min(r["T_R"] for r in rounds),
                            "set_size_75": len(sets[0.25][0]),
                            "set_size_90": len(sets[0.10][0]),
                            "order": " > ".join(names[r["eliminated"]] for r in rounds)})
    for i in range(M):
        for j in range(i):
            d = L_port[:, i] - L_port[:, j]
            dd = d - d.mean()
            v = np.dot(dd, dd) / T
            for g in range(1, lag + 1):
                v += 2 * (1 - g / (lag + 1)) * np.dot(dd[g:], dd[:-g]) / T
            t = d.mean() / np.sqrt(max(v, 1e-30) / T)
            dm_rows.append({"trade_lag": align, "row": names[i], "col": names[j], "DM": t,
                            "p": 2 * (1 - sp_stats.norm.cdf(abs(t)))})
port = pd.DataFrame(rows)
port.to_csv(OUT / "Table4_MCS_portfolio.csv", index=False)
mg = pd.DataFrame(margin_rows)
dm = pd.DataFrame(dm_rows)

wl = pd.read_csv(OUT / "Table4_MCS_blocklen.csv")
wl5 = wl[(wl.block_len == 5) & (wl["loss"] == "MSE")].set_index("model")
wl20 = wl[(wl.block_len == 20) & (wl["loss"] == "MSE")].set_index("model")
pA = pd.read_csv(OUT / "Table4_PanelA.csv").set_index("model")

summary = []
for align, net in net_by.items():
    p5 = port[(port["trade_lag"] == align) & (port.block_len == 5)].set_index("model")
    for nm in names:
        summary.append({
            "trade_lag": align, "model": nm,
            "ann_return": net[nm].mean() * 252,
            "ann_sharpe_net": net[nm].mean() / net[nm].std(ddof=1) * np.sqrt(252),
            "mean_daily_turnover": tno_by[align][nm].mean(),
            "MCS_p_weightloss": wl5.MCS_pval.loc[nm],
            "MCS_p_portfolio": p5.MCS_pval.loc[nm],
            "in75_portfolio": bool(p5.in_MCS_75.loc[nm]),
        })
sm = pd.DataFrame(summary)
sm.to_csv(OUT / "Table4_MCS_portfolio_summary.csv", index=False)

pd.set_option("display.width", 240)
print(f"\n=== MCS 推断口径: B={mcs.N_BOOTSTRAP} seed={mcs.SEED} 块长={mcs.BLOCK_LENGTHS} "
      f"NW-HAC 滞后={lag} ===")
for align in net_by:
    lo, me, hi = corr_stat[align]
    print(f"\n--- 成交口径 {align}: Y[t] 赚 {'t' if align == 'same' else 't+1'} 日收益 ---")
    print(sm[sm["trade_lag"] == align].drop(columns=["trade_lag"]).set_index("model").to_string())
    print(mg[mg["trade_lag"] == align][["block_len", "max_p_hat", "min_T_R", "set_size_75",
                                 "set_size_90"]].to_string(index=False))
    for _, m in mg[mg["trade_lag"] == align].iterrows():
        print(f"  l={m['block_len']:>2}: {m['order']}")
    d = dm[dm["trade_lag"] == align]
    print(f"  模型间日损失相关 min/中位/max = {lo:.3f}/{me:.3f}/{hi:.3f}")
    print(f"  NW-HAC DM 21 对: p<0.10 有 {(d.p < 0.10).sum()} 对, p<0.05 有 "
          f"{(d.p < 0.05).sum()} 对, 最大|DM|={d.DM.abs().max():.2f}")
    print(d[d.p < 0.10].drop(columns="trade_lag").to_string(index=False))

# ------------------------------------------------------------- LaTeX 并排表
FLOOR = 1 / (mcs.N_BOOTSTRAP + 1)
MAIN = "same"
p5 = port[(port["trade_lag"] == MAIN) & (port.block_len == 5)].set_index("model")
p20 = port[(port["trade_lag"] == MAIN) & (port.block_len == 20)].set_index("model")
s5 = sm[(sm["trade_lag"] == MAIN)].set_index("model")
p5n = port[(port["trade_lag"] == "next") & (port.block_len == 5)].set_index("model")


def ptxt(p):
    return f"{p:.3f}" if p >= 10 * FLOOR else f"{FLOOR:.4f}"


def mkcell(frame, model):
    r = frame.loc[model]
    fl = "**" if r.in_MCS_75 else ("*" if r.in_MCS_90 else "")
    return f"{ptxt(r.MCS_pval)}$^{{{fl}}}$" if fl else ptxt(r.MCS_pval)


def bb(x, on):
    return f"\\textbf{{{x}}}" if on else str(x)


A = []
A.append("\\begin{table}[htbp]\n\\centering\n\\begin{threeparttable}")
A.append("\\caption{Model confidence set under weight-tracking and portfolio losses}"
         "\\label{tab:mcsloss}")
A.append("\\begin{tabular}{lccccccc}\n\\toprule")
A.append("& \\multicolumn{3}{c}{Squared weight loss} & "
         "\\multicolumn{4}{c}{Portfolio net-return loss} \\\\")
A.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}")
A.append("Model & MSE$_w$ ($10^{-5}$) & $p$ ($l$=5) & $p$ ($l$=20) "
         "& Ann.\\ ret.\\ (\\%) & Net Sharpe & $p$ ($l$=5) & $p$ ($l$=20) \\\\")
A.append("\\midrule")
for i, nm in enumerate(names):
    h = nm == "Network VARX"
    A.append(" & ".join([
        f"({i+1}) {bb(nm, h)}",
        bb(f"{pA.MSE_w.loc[nm] * 1e5:.2f}", h),
        mkcell(wl5, nm), mkcell(wl20, nm),
        f"{s5['ann_return'].loc[nm] * 100:.1f}",
        f"{s5['ann_sharpe_net'].loc[nm]:.2f}",
        mkcell(p5, nm), mkcell(p20, nm),
    ]) + " \\\\")
A.append("\\bottomrule\n\\end{tabular}")
A.append("\\begin{tablenotes}[flushleft]\\footnotesize")
A.append("\\item Both blocks use the same $T_R$ statistic of \\cite{HansenLundeNason2011}, the "
         "same moving block bootstrap with $l$ trading days, 2{,}000 replications and a common "
         "seed, on the same 200 evaluation dates and the same seven direct models. Only the loss "
         "changes: squared weight error averaged over the assets, versus the negative of the daily "
         "portfolio return net of a one-way cost of $1$ basis point on economic turnover. "
         "$^{*}$ and $^{**}$ mark membership of the 90\\% and 75\\% set.")
pb = pd.read_csv(OUT / "Table4_PanelB_DM_squared.csv")
n_wl_sig = int((pb.p < 0.10).sum())
A.append("\\item Weight-loss levels of $" + f"{FLOOR:.4f}" + "$ sit at the bootstrap resolution "
         "floor $1/(B+1)$: that test rejects in every round, so its set is a singleton at every "
         "block length. The portfolio-loss test rejects once, removing model (4), and stops in "
         "the second round with six models at the 75\\% level and all seven at the 90\\% level; "
         f"only {int((dm[(dm['trade_lag'] == MAIN)].p < 0.10).sum())} of 21 pairwise HAC "
         f"comparisons reach $p<0.10$, against {n_wl_sig} of 21 under squared weight loss. The "
         "ranking also inverts: the model with the lowest weight loss is the only one outside the "
         "75\\% portfolio set, while the drift-corrected variant attains the highest net Sharpe "
         "ratio.")
n_next75 = int(port[(port["trade_lag"] == "next") & (port.block_len == 5)].in_MCS_75.sum())
A.append("\\item Portfolio returns assume the forecast weight is traded on the day it forecasts. "
         "Forming instead one day later, as in Table~\\ref{tab:table3}, leaves the 75\\% set at "
         f"{n_next75} models "
         "and gives the ranking reported in the text.\n\\end{tablenotes}")
A.append("\\end{threeparttable}\n\\end{table}")
(OUT / "Table4_MCS_portfolio_LaTeX.tex").write_text("\n".join(A), encoding="utf-8")
print("\nLaTeX ->", OUT / "Table4_MCS_portfolio_LaTeX.tex")
