import urllib.request, pandas as pd

# 1. 从 CBOE 下载全部历史数据
resp = urllib.request.urlopen(
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
df = pd.read_csv(resp)

# 2. 日期格式化为 YYYYMMDD，截取 2010-06 ~ 2020-05
df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y%m%d")
df = df[df["DATE"].between("20100601", "20200501")]

# 3. 保留收盘价，存盘
df = df[["DATE", "CLOSE"]].rename(columns={"DATE": "date", "CLOSE": "vix"})
df = df.set_index("date")
df.to_csv(r"d:\HuaweiMoveData\Users\27438\Desktop\大创\数据\vix_daily.csv")
