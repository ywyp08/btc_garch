import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# read the data
btc = pd.read_csv("data/BTC_weekly.csv")
eth = pd.read_csv("data/ETH_daily.csv")
usdt = pd.read_csv("data/USDt_daily.csv")

# select columns
btc_date = pd.to_datetime(btc.iloc[:, 0])
btc_open = pd.to_numeric(btc.iloc[:, 2].str.replace(',', ''))
btc_open_log = np.log(btc_open)
btc_change = pd.to_numeric(btc.iloc[:, 6].str.replace('%', ''))

eth_date = pd.to_datetime(eth.iloc[:, 0])
eth_open = pd.to_numeric(eth.iloc[:, 2].str.replace(',', ''))
eth_change = pd.to_numeric(eth.iloc[:, 6].str.replace('%', ''))

usdt_date = pd.to_datetime(usdt.iloc[:, 0])
usdt_change = pd.to_numeric(usdt.iloc[:, 6].str.replace('%', ''))

# model
model = arch_model(btc_change, p=1, q=3)
model_fit = model.fit()
print(model_fit.summary())

# predictions
predictions = model_fit.forecast(start=0)

# plot
ffig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(btc_date, np.log(predictions.variance['h.1']), 'g-')
ax2.plot(btc_date, btc_open_log, 'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Predictions', color='g')
ax2.set_ylabel('Open', color='b')
#plot_pacf(btc_change**2)
plt.show()