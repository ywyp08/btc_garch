import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# read the data
btc = pd.read_csv("../data/BTC.csv")
eth = pd.read_csv("../data/ETH.csv")
usdt = pd.read_csv("../data/USDt.csv")

# select columns
btc_date = pd.to_datetime(btc.iloc[:, 0])
btc_open = pd.to_numeric(btc.iloc[:, 2].str.replace(',', ''))
btc_change = pd.to_numeric(btc.iloc[:, 6].str.replace('%', ''))

usdt_date = pd.to_datetime(usdt.iloc[:, 0])
usdt_change = pd.to_numeric(usdt.iloc[:, 6].str.replace('%', ''))

eth_date = pd.to_datetime(eth.iloc[:, 0])
eth_change = pd.to_numeric(eth.iloc[:, 6].str.replace('%', ''))

# slice the data - last year
usdt_date_year = usdt_date[:365]
usdt_change_year = usdt_change[:365]

# plot the data
plt.plot(usdt_date_year, usdt_change_year)
plt.show()

# model
model = arch_model(btc_change, p=2, q=2)
model_fit = model.fit()
print(model_fit.summary())
