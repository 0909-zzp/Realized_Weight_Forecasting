import numpy as np, sys, time
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\HuaweiMoveData\Users\27438\Desktop\大创')
from mcs_pvalues import mcs_pvalues

base = r'D:\HuaweiMoveData\Users\27438\Desktop\大创'
Y = np.load(base + r'\特征工程\Y_targets.npy')
Yte = Y[-363:]
T0, T1 = 65, 265
Yte_eval = Yte[T0:T1]

names = ['VAR','Sparse VAR','Sparse VARX','Network VARX','Network+Smooth','DFL','LSTM']
L_mse = np.empty((T1-T0, 7)); L_mae = np.empty((T1-T0, 7))
for m in range(1,8):
    P = np.load(base + rf'\VARX\Y_pred_model{m}.npy')
    E = P[T0:T1] - Yte_eval
    L_mse[:, m-1] = (E**2).mean(axis=1)
    L_mae[:, m-1] = np.abs(E).mean(axis=1)

rows = []
for stat in ['Tmax','TR']:
    for label, L in [('MSE', L_mse), ('MAE', L_mae)]:
        pvals, surv = mcs_pvalues(L, alpha=0.10, B=9999, statistic=stat, seed=42)
        for i,n in enumerate(names):
            rows.append(dict(model=n, loss=label, stat=stat, MCS_p=pvals[i], in_MCS=i in surv))

df = pd.DataFrame(rows)
print(df[df.stat=='TR'].pivot(index='model', columns='loss', values='MCS_p').to_string())
out = base + r'\Table4_MCS_pvalues.csv'
# 最终表: 每个模型一行, MSE 和 MAE 的 MCS p 值(TR 统计量)
final = df[(df.stat=='TR')].pivot(index='model', columns='loss', values='MCS_p').reset_index()
final.columns = ['Model','MCS_p_MSE','MCS_p_MAE']
# 保留4位小数, 但0.0001按4位显示0.0001
final.to_csv(out, index=False)
print('saved:', out)
print(final.to_string(index=False))
