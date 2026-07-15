"""下载 FRED 宏观指标 CSV"""
import urllib.request, pandas as pd

D = r"d:\HuaweiMoveData\Users\27438\Desktop\大创\数据"
SERIES = {
    "term_spread.csv": "T10Y2Y",
    "credit_spread.csv": "BAA10YM",
    "dxy_close.csv": "DTWEXBGS",
}

for fname, sid in SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    urllib.request.urlretrieve(url, f"{D}/{fname}")
    df = pd.read_csv(f"{D}/{fname}")
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.strftime("%Y%m%d")
    df.to_csv(f"{D}/{fname}", index=False)
    print(f"{fname}: {len(df)} rows, {df['observation_date'].min()}~{df['observation_date'].max()}")
