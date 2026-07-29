"""LSTM GMVP 权重预测 — 独立版
纯 NumPy, 单层 LSTM(128) + softmax(T=0.5) + dropout(0.05)
用法: 从 Table2 主脚本调用, 或单独 import 训练.
"""
import numpy as np, time

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
d_tanh    = lambda z, a: 1 - a**2
d_sigmoid = lambda a: a * (1 - a)

def build_lstm(K=392, n_feat=1185, hid=128, seq_len=30,
               lr=0.0005, dropout_rate=0.05, softmax_T=0.5, seed=42):
    """构建 LSTM 的 forward 和 step 函数, 返回 (forward, step, params)."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0/(n_feat+hid))
    W = {g: scale*rng.standard_normal((hid,n_feat+hid)).astype(np.float64) for g in 'f i o c'.split()}
    b = {g: (np.ones(hid) if g=='f' else np.zeros(hid)).astype(np.float64) for g in 'f i o c'.split()}
    Wy = np.sqrt(2.0/(K+hid))*rng.standard_normal((K,hid)).astype(np.float64)
    by = np.zeros(K, dtype=np.float64)

    M, V = {}, {}
    for g in 'f i o c'.split():
        M[f'W_{g}'], V[f'W_{g}'] = np.zeros_like(W[g]), np.zeros_like(W[g])
        M[f'b_{g}'], V[f'b_{g}'] = np.zeros_like(b[g]), np.zeros_like(b[g])
    M['Wy'], V['Wy'] = np.zeros_like(Wy), np.zeros_like(Wy)
    M['by'], V['by'] = np.zeros_like(by), np.zeros_like(by)

    def forward(X_batch, training=True):
        B = len(X_batch); h = np.zeros((B,hid)); c = np.zeros((B,hid))
        cache = []
        for t in range(seq_len):
            x_t = X_batch[:,t,:]; hp, cp = h.copy(), c.copy()
            xh = np.hstack([x_t, h])
            i = sigmoid(xh@W['i'].T + b['i'])
            f = sigmoid(xh@W['f'].T + b['f'])
            o = sigmoid(xh@W['o'].T + b['o'])
            ct = np.tanh(xh@W['c'].T + b['c'])
            c = f*c + i*ct; h = o*np.tanh(c)
            cache.append((hp, cp, i, f, o, ct, x_t))
        mask = None
        if training and dropout_rate > 0:
            mask = (rng.random(h.shape) > dropout_rate).astype(np.float64)/(1-dropout_rate)
            h = h*mask
        logits = h@Wy.T + by
        logits = logits/softmax_T - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        y_pred = exp_l/exp_l.sum(axis=1, keepdims=True)
        return y_pred, cache, mask, logits, h, c  # +c for BPTT

    def step(X_batch, Y_batch, t_step):
        nonlocal Wy, by, W, b
        B = len(X_batch)
        y_pred, cache, mask, logits, h_out, c_out = forward(X_batch, training=True)
        diff = y_pred - Y_batch; loss_val = np.mean(diff**2)
        dy = (2.0/(B*K))*diff
        dlogits = dy*y_pred*(1.0-y_pred)/softmax_T
        dWy = dlogits.T@h_out; dby = dlogits.sum(axis=0)
        dh = dlogits@Wy
        if mask is not None: dh = dh*mask
        dc = np.zeros((B,hid))
        dW_g = {g: np.zeros_like(W[g]) for g in 'f i o c'.split()}
        db_g = {g: np.zeros_like(b[g]) for g in 'f i o c'.split()}
        for t in range(seq_len-1, -1, -1):
            hp, cp, ig, fg, og, ctg, x_t = cache[t]
            c_cur = cache[t+1][1] if t < seq_len-1 else c_out  # BPTT fix: c(t) stored in next cp
            h_cur = cache[t+1][0] if t < seq_len-1 else h_out
            do = dh*np.tanh(c_cur)
            dc = dc + dh*og*d_tanh(None, np.tanh(c_cur))
            di = dc*ctg; dct = dc*ig; df = dc*cp
            do_g = do*d_sigmoid(og); di_g = di*d_sigmoid(ig)
            df_g = df*d_sigmoid(fg); dct_g = dct*d_tanh(None, ctg)
            xh = np.hstack([x_t, hp])
            for gk, gv in [('o',do_g),('i',di_g),('f',df_g),('c',dct_g)]:
                dW_g[gk] += gv.T@xh; db_g[gk] += gv.sum(axis=0)
            dh_xh = do_g@W['o']+di_g@W['i']+df_g@W['f']+dct_g@W['c']
            dh = dh_xh[:, n_feat:]; dc = dc*fg
        for name, dval in [('Wy', dWy), ('by', dby)] + \
            [(f'W_{g}', dW_g[g]) for g in 'f i o c'.split()] + \
            [(f'b_{g}', db_g[g]) for g in 'f i o c'.split()]:
            dval = np.clip(dval, -1.0, 1.0)
            M[name] = 0.9*M[name] + 0.1*dval
            V[name] = 0.999*V[name] + 0.001*dval**2
            m_hat = M[name]/(1.0-0.9**(t_step+1))
            v_hat = V[name]/(1.0-0.999**(t_step+1))
            upd = lr*m_hat/(np.sqrt(v_hat)+1e-8)
            if name == 'Wy': Wy -= upd
            elif name == 'by': by -= upd
            elif name.startswith('W_'): W[name[2]] -= upd
            else: b[name[2]] -= upd
        return loss_val

    params = (lr, seq_len, epochs=500, patience=50, batch=64)
    return forward, step, params


def make_sequences(X, Y, seq_len):
    n = len(X); xs, ys = [], []
    for i in range(seq_len-1, n):
        xs.append(X[i-seq_len+1:i+1]); ys.append(Y[i])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def train(forward, step, X_tr, Y_tr, X_val, Y_val, seq_len,
          batch=64, epochs=500, patience=50, verbose=True):
    Xs_tr, Ys_tr = make_sequences(X_tr, Y_tr, seq_len)
    Xs_val, Ys_val = make_sequences(X_val, Y_val, seq_len) if len(Y_val)>=seq_len else (Xs_tr[:1], Ys_tr[:1])
    n_batches = max(1, len(Xs_tr)//batch)
    best_val, wait = np.inf, 0; t0 = time.time()
    for ep in range(epochs):
        idx = np.random.permutation(len(Xs_tr)); losses = []
        for ib in range(n_batches):
            bi = idx[ib*batch:(ib+1)*batch]
            losses.append(step(Xs_tr[bi], Ys_tr[bi], ep*n_batches+ib))
        yv, _, _, _, _, _ = forward(Xs_val, training=False)
        val_mse = float(np.mean((yv-Ys_val)**2))
        if val_mse < best_val: best_val, wait = val_mse, 0
        else: wait += 1
        if verbose and ep%20 == 0:
            print(f"  ep {ep:3d}: train={np.mean(losses):.4e}  val={val_mse:.4e}  best={best_val:.4e}")
        if wait >= patience: break
    print(f"  训练完成 ({time.time()-t0:.0f}s), 最佳验证MSE={best_val:.4e}")
    return best_val


def predict(forward, X_te, seq_len, pad):
    Xs_te, _ = make_sequences(X_te, np.zeros((len(X_te), pad.shape[1])), seq_len)
    yp, _, _, _, _, _ = forward(Xs_te, training=False)
    result = np.vstack([pad[:seq_len-1], yp])
    s = result.sum(1, keepdims=True); s = np.where(np.abs(s)<1e-10, 1.0, s)
    return result/s


if __name__ == "__main__":
    print("LSTM 独立模块. 用法:")
    print("  fwd, step, params = build_lstm(K=392, n_feat=1185)")
    print("  train(fwd, step, X_tr, Y_tr, X_val, Y_val, seq_len=30)")
    print("  Y_pred = predict(fwd, X_te, 30, pad=M4_predictions)")
