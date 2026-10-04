import numpy as np
import pandas as pd
import sys

def stationary_bootstrap_indices(T, mean_block, B, rng):
    """Stationary bootstrap (Politis & Romano, 1994): geometric block lengths."""
    idx = np.empty((B, T), dtype=int)
    p = 1.0 / mean_block
    for b in range(B):
        s = 0
        while s < T:
            length = rng.geometric(p)
            start = rng.integers(0, T)
            for j in range(length):
                if s >= T:
                    break
                idx[b, s] = (start + j) % T
                s += 1
    return idx


def mcs_pvalues(L, alpha=0.10, B=9999, mean_block=None,
                statistic="Tmax", seed=12345):
    """
    Hansen-Lunde-Nason (2011) MCS p-values.

    Parameters
    ----------
    L : (T, m) array, loss series. Rows = dates (common valid dates),
        columns = competing models. Smaller loss is better.
    alpha : elimination significance level (e.g. 0.10 for a 90% MCS).
    B : number of bootstrap replications (use >= 9999).
    mean_block : average block length for the stationary bootstrap.
        Default: max(3, ceil(T**(1/3))). Ideally chosen by AR(p) fit on
        the loss-difference series as in Hansen et al. (2011).
    statistic : "Tmax" (deviation from the set average) or "TR" (pairwise range).

    Returns
    -------
    pvals : (m,) MCS p-values. Eliminated models get the bootstrap p-value at
        the step they were eliminated (cumulative max). Final surviving models
        get 1.0 by definition.
    survivors : list of model indices in the MCS at level alpha.
    """
    rng = np.random.default_rng(seed)
    T, m = L.shape
    if T < 10:
        raise ValueError("need more than 10 time observations for the block bootstrap")
    if mean_block is None:
        mean_block = max(3, int(np.ceil(T ** (1.0 / 3.0))))
    if mean_block > T:
        mean_block = T

    idx = stationary_bootstrap_indices(T, mean_block, B, rng)

    alive = list(range(m))
    pvals = np.zeros(m)
    running = 0.0

    while len(alive) > 1:
        M = alive
        # losses centered on the current set average (relative losses)
        C = L[:, M] - L[:, M].mean(axis=1, keepdims=True)
        dbar = C.mean(axis=0)

        # bootstrap distribution of the average relative losses
        dbar_b = np.empty((B, len(M)))
        for b in range(B):
            dbar_b[b] = C[idx[b], :].mean(axis=0)
        se = dbar_b.std(axis=0, ddof=1)
        se[se < 1e-15] = 1e-15

        if statistic == "Tmax":
            t = dbar / se
            Tobs = np.max(np.abs(t))
            Tb = np.max(np.abs((dbar_b - dbar) / se), axis=1)
        else:  # TR, pairwise range statistic
            dbar_ij = dbar[:, None] - dbar[None, :]
            diffs_b = dbar_b[:, :, None] - dbar_b[:, None, :]
            var_ij = diffs_b.var(axis=0, ddof=1)
            var_ij[var_ij < 1e-15] = 1e-15
            se_ij = np.sqrt(var_ij)
            tmat = dbar_ij / se_ij
            np.fill_diagonal(tmat, 0.0)
            Tobs = np.max(np.abs(tmat))
            Tb = np.array([
                np.max(np.abs((diffs_b[b] - dbar_ij) / se_ij))
                for b in range(B)
            ])

        # finite-sample bootstrap p-value: never exactly 0
        p = (1.0 + np.sum(Tb >= Tobs)) / (B + 1.0)
        running = max(running, p)

        if p <= alpha:
            worst = int(np.argmax(dbar))       # largest average relative loss
            model = M[worst]
            pvals[model] = running             # max p over the steps it survived
            alive.pop(worst)
        else:
            break

    # HLN definition: models that remain in the final set receive p-value 1.
    for model in alive:
        pvals[model] = 1.0

    return pvals, alive


def main():
    if len(sys.argv) < 2:
        print("usage: python mcs_pvalues.py losses.csv [alpha] [B] [Tmax|TR]")
        print("  losses.csv: rows = dates, columns = models, header = model names")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)
    models = list(df.columns)
    L = df.to_numpy(float)

    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 9999
    stat = sys.argv[4] if len(sys.argv) > 4 else "Tmax"

    pvals, survivors = mcs_pvalues(L, alpha=alpha, B=B, statistic=stat)
    out = pd.DataFrame({"model": models, "MCS_p": pvals})
    out["in_MCS"] = out["model"].isin([models[i] for i in survivors])
    out = out.sort_values("MCS_p", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
