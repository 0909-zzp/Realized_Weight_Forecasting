"""Table 4 Panel A 的 MCS: α 停止规则 + 块长敏感性 (HLN 2011 附录 p6/p9/p12 式做法)。

不重跑模型、不动 Panel B/C —— 只重算 MCS 推断, 并接管 Panel A 的 LaTeX 输出。
输出:
  Table4_MCS_blocklen.csv   每模型 × 每损失 × 每块长 的 p 值与成员资格
  Table4_MCS_rounds.csv     每轮 T_R / p_hat / 淘汰对象, 供 Appendix B 引用
  Table4_LaTeX.tex          Panel A (规范版本)
  Table4_MCS_appendix_LaTeX.tex   块长敏感性表
"""
import os as _os
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

PROJ = Path(r"D:\HuaweiMoveData\Users\27438\Desktop\大创")
OUT = PROJ / "性能评估与可视化"
VARX = PROJ / "VARX"

spec = importlib.util.spec_from_file_location("mcs", str(OUT / "MCS_完整程序.py"))
mcs = importlib.util.module_from_spec(spec)
sys.modules["mcs"] = mcs
spec.loader.exec_module(mcs)

names = ["VAR", "Sparse VAR", "Sparse VARX", "Network VARX",
         "Network VARX-S", "Network VARX-S-DF", "LSTM"]
mcs.MODEL_NAMES = {i + 1: names[i] for i in range(len(names))}

n = len(np.load(PROJ / "特征工程" / "X_features.npy"))
test_start = int(n * .7) + int(n * .15)
Y = np.load(PROJ / "特征工程" / "Y_targets.npy")[test_start:][65:265]
files = ["Y_pred_model1.npy", "Y_pred_model2.npy", "Y_pred_model3.npy",
         "Y_pred_model4.npy", "Y_pred_model5.npy", "Y_pred_model6_opt.npy",
         "Y_pred_model7.npy"]
preds = [np.load(VARX / f)[-363:][65:265] for f in files]
losses = {
    "MSE": np.column_stack([((p - Y) ** 2).mean(axis=1) for p in preds]),
    "MAE": np.column_stack([np.abs(p - Y).mean(axis=1) for p in preds]),
}
T, M = losses["MSE"].shape
print(f"T={T}天  K={Y.shape[1]}资产  M={M}模型  B={mcs.N_BOOTSTRAP}  seed={mcs.SEED}")


def final_round_margin(L, blk, n_boot, seed):
    """末轮(幸存模型 vs 次优)的 T_R 及其 block-bootstrap 临界值。"""
    bi = mcs.block_bootstrap(L.shape[0], n_boot, blk, np.random.default_rng(seed))
    rounds, _ = mcs.mcs_path(L, n_boot, blk, seed, verbose=False)
    sub = L[:, rounds[-1]["models_before"]]
    dbar = sub.mean(0)
    D = dbar[:, None] - dbar[None, :]
    bm = sub[bi, :].mean(axis=1)
    Db = bm[:, :, None] - bm[:, None, :]
    sd = np.sqrt(np.maximum(((Db - D) ** 2).mean(axis=0), 1e-20))
    Tobs = float(np.abs(D / sd).max())
    Tboot = np.abs((Db - D) / sd).max(axis=(1, 2))
    return (Tobs, float(np.quantile(Tboot, 0.90)), float(np.quantile(Tboot, 0.75)),
            [names[i] for i in rounds[-1]["models_before"]])


rows, round_rows, margin_rows = [], [], []
for lname, L in losses.items():
    for blk in mcs.BLOCK_LENGTHS:
        rounds, last = mcs.mcs_path(L, mcs.N_BOOTSTRAP, blk, mcs.SEED, verbose=False)
        sets = mcs.mcs_member_sets(rounds, M)
        for r in rounds:
            round_rows.append({
                "loss": lname, "block_len": blk, "round": r["round"],
                "n_models": len(r["models_before"]), "T_R": r["T_R"],
                "p_hat": r["p_hat"], "eliminated": names[r["eliminated"]],
                "t_worst": round(r["t_worst"], 3),
                "stop_round_75": sets[0.25][1], "stop_round_90": sets[0.10][1],
            })
        for i, nm in enumerate(names):
            rec = next((r for r in rounds if r["eliminated"] == i), None)
            rows.append({
                "loss": lname, "block_len": blk, "model": nm,
                "elim_round": -1 if rec is None else rec["round"],
                "MCS_pval": 1.0 if rec is None else rec["p_hat"],
                "in_MCS_75": i in sets[0.25][0], "in_MCS_90": i in sets[0.10][0],
                "set_size_75": len(sets[0.25][0]), "set_size_90": len(sets[0.10][0]),
            })
        TR, c90, c75, pair = final_round_margin(L, blk, mcs.N_BOOTSTRAP, mcs.SEED)
        margin_rows.append({"loss": lname, "block_len": blk, "T_R_final": TR,
                            "crit_90": c90, "crit_75": c75,
                            "final_pair": " vs ".join(pair),
                            "max_p_hat": max(r["p_hat"] for r in rounds),
                            "set_size": len(sets[0.25][0])})

df = pd.DataFrame(rows)
rr = pd.DataFrame(round_rows)
mg = pd.DataFrame(margin_rows)
df.to_csv(OUT / "Table4_MCS_blocklen.csv", index=False)
rr.to_csv(OUT / "Table4_MCS_rounds.csv", index=False)
mg.to_csv(OUT / "Table4_MCS_margins.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== 每损失×每块长: 各轮最大 p_hat、末轮 T_R 与临界值、集合大小 ===")
print(mg.to_string(index=False))
print("\n=== 淘汰顺序 (R1→末轮) ===")
for (ln, blk), g in rr.groupby(["loss", "block_len"]):
    print(f"  {ln} block={blk:>2}: " + " > ".join(g.eliminated))
print("\n=== 模型级 MCS p 值 ===")
print(df.pivot_table(index="model", columns=["loss", "block_len"], values="MCS_pval").to_string())

# ---------------------------------------------------------------- LaTeX: Panel A
BLK = 5
pa = df[df.block_len == BLK].set_index(["loss", "model"])
mse_w = losses["MSE"].mean(axis=0)
mae_w = losses["MAE"].mean(axis=0)
min_p = 1.0 / (mcs.N_BOOTSTRAP + 1)
bfmt = f"{mcs.N_BOOTSTRAP:,}".replace(",", "{,}")
TR_min_headline = float(rr.loc[rr.block_len == BLK, "T_R"].min())
TR_min_grid = float(rr["T_R"].min())
c90_max = float(mg["crit_90"].max())


def cell(loss, model):
    p = pa.loc[(loss, model), "MCS_pval"]
    flag = "**" if pa.loc[(loss, model), "in_MCS_75"] else ("*" if pa.loc[(loss, model), "in_MCS_90"] else "")
    txt = f"{p:.2f}" if p >= 0.005 else "0.00"
    return f"{txt}$^{{{flag}}}$" if flag else txt


L = []
L.append("\\begin{table}[htbp]\n\\centering\n\\begin{threeparttable}")
L.append("\\caption{Weight forecast accuracy and model confidence set}\\label{tab:table4A}")
L.append("\\begin{tabular}{lcccc}\n\\toprule")
L.append("Model & MSE$_w$ ($10^{-5}$) & MAE$_w$ ($10^{-3}$) & MCS $p$ (MSE) & MCS $p$ (MAE) \\\\\n\\midrule")
for i, nm in enumerate(names):
    b = "\\textbf" if nm == "Network VARX" else ""
    L.append(f"({i+1}) {b}{{{nm}}} & {b}{{{mse_w[i]*1e5:.2f}}} & {b}{{{mae_w[i]*1e3:.2f}}} "
             f"& {cell('MSE', nm)} & {cell('MAE', nm)} \\\\")
L.append("\\bottomrule\n\\end{tabular}")
L.append("\\begin{tablenotes}[flushleft]\\footnotesize")
L.append(f"\\item Panel A. Daily losses averaged over the $K={Y.shape[1]}$ assets on the "
         f"{T} common evaluation dates. MCS $p$-values follow \\cite{{HansenLundeNason2011}}: the level "
         "reported for a model is that of the test of equal predictive ability in the round "
         "in which it is eliminated, monotonized along the elimination sequence. "
         "$^{*}$ and $^{**}$ denote membership of the 90\\% and 75\\% confidence set.")
L.append(f"\\item The set is a singleton, model (4), at both confidence levels and under both "
         f"losses, and the sequential procedure never stops early: the smallest $T_R$ statistic "
         f"over all elimination rounds is {TR_min_headline:.2f} at $l={BLK}$, and still "
         f"{TR_min_grid:.2f} at the longest block length considered, against a block-bootstrap "
         f"90\\% critical value of about {c90_max:.2f} "
         f"($B={bfmt}$ replications, seed {mcs.SEED}). A level reported as 0.00 is "
         f"the bootstrap resolution floor of $1/(B+1)={min_p:.4f}$, not an infinitely large gap.")
r_var = int(rr.loc[(rr.block_len == BLK) & (rr["loss"] == "MSE") &
                   (rr.eliminated == "VAR"), "round"].iloc[0])
L.append(f"\\item Elimination order records which model is most distinguishable from the rest, not which "
         f"carries the largest loss: under squared loss model (1) has by far the highest average "
         f"loss yet exits only in round {r_var}. Round numbers refer to $l={BLK}$; under absolute loss the "
         "exit order is unchanged across block lengths, whereas under squared loss the order "
         "within the dominated group is not. The surviving model and the runner-up removed in the "
         "final round are invariant throughout; Appendix Table~\\ref{tab:mcsblock} reports the "
         "grid over $l\\in\\{5,10,20\\}$.")
L.append("\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}")
(OUT / "Table4_LaTeX.tex").write_text("\n".join(L), encoding="utf-8")
print("\nPanel A LaTeX ->", OUT / "Table4_LaTeX.tex")

# ---------------------------------------------------------- LaTeX: 附录块长表
A = []
A.append("\\begin{table}[htbp]\n\\centering\n\\begin{threeparttable}")
A.append("\\caption{Model confidence set: sensitivity to the block-bootstrap length}"
         "\\label{tab:mcsblock}")
A.append("\\begin{tabular}{lcccccc}\n\\toprule")
A.append("& \\multicolumn{3}{c}{Squared weight loss} & \\multicolumn{3}{c}{Absolute weight loss} \\\\")
A.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
A.append("Block $l$ & $\\max_n \\hat{p}_n$ & $T_R$ (final) & $c_{90}$ & $\\max_n \\hat{p}_n$ & $T_R$ (final) & $c_{90}$ \\\\\n\\midrule")
for blk in mcs.BLOCK_LENGTHS:
    g = mg[mg.block_len == blk].set_index("loss")
    A.append(f"{blk} & {g.loc['MSE','max_p_hat']:.4f} & {g.loc['MSE','T_R_final']:.2f} & "
             f"{g.loc['MSE','crit_90']:.2f} & {g.loc['MAE','max_p_hat']:.4f} & "
             f"{g.loc['MAE','T_R_final']:.2f} & {g.loc['MAE','crit_90']:.2f} \\\\")
A.append("\\bottomrule\n\\end{tabular}")
A.append("\\begin{tablenotes}[flushleft]\\footnotesize\n\\item $\\max_n\\hat{p}_n$ is the largest "
         "elimination-round significance level over all six rounds, $T_R$ (final) is the statistic "
         "in the last round, which compares the surviving model with the runner-up, and $c_{90}$ is "
         "the 90th percentile of its moving-block-bootstrap null distribution. Because "
         "$\\max_n\\hat{p}_n$ stays far below both $\\alpha=0.10$ and $\\alpha=0.25$, the "
         "90\\% and 75\\% sets coincide and contain one model in every cell. The final round is "
         "the binding case: earlier rounds separate the surviving model from a wider set by larger "
         "statistics.\n\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}")
(OUT / "Table4_MCS_appendix_LaTeX.tex").write_text("\n".join(A), encoding="utf-8")
print("附录块长表 ->", OUT / "Table4_MCS_appendix_LaTeX.tex")
