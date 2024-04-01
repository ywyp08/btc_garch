import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Data processing
btc = pd.read_csv("data/BTC_daily.csv")
eth = pd.read_csv("data/ETH_daily.csv")
usdt = pd.read_csv("data/USDt_daily.csv")
gold = pd.read_csv("data/Gold_daily.csv")

btc_date = pd.to_datetime(btc.iloc[:, 0])
btc_open = pd.to_numeric(btc.iloc[:, 2].str.replace(',', ''))
btc_open_log = np.log(btc_open)
btc_return = pd.to_numeric(btc.iloc[:, 6].str.replace('%', ''))

eth_date = pd.to_datetime(eth.iloc[:, 0])
eth_open = pd.to_numeric(eth.iloc[:, 2].str.replace(',', ''))
eth_open_log = np.log(eth_open)
eth_return = pd.to_numeric(eth.iloc[:, 6].str.replace('%', ''))

usdt_date = pd.to_datetime(usdt.iloc[:, 0])
usdt_return = pd.to_numeric(usdt.iloc[:, 6].str.replace('%', ''))

gold_date = pd.to_datetime(gold.iloc[:, 0])
gold_return = pd.to_numeric(gold.iloc[:, 6].str.replace('%', ''))


# Modely
btc_garch44 = arch_model(btc_return, mean='Constant', vol='GARCH', p=4, q=4)
btc_garch44_fit = btc_garch44.fit()
btc_egarch44 = arch_model(btc_return, mean='Constant', vol='EGARCH', p=4, q=4)
btc_egarch44_fit = btc_egarch44.fit()

btc_garch14 = arch_model(btc_return, mean='Constant', vol='GARCH', p=1, q=4)
btc_garch14_fit = btc_garch14.fit()
btc_egarch11 = arch_model(btc_return, mean='Constant', vol='EGARCH', p=1, q=1) # vice moznosti
btc_egarch11_fit = btc_egarch11.fit()

eth_garch55 = arch_model(eth_return, mean='Constant', vol='GARCH', p=5, q=5)
eth_garch55_fit = eth_garch55.fit()
eth_egarch55 = arch_model(eth_return, mean='Constant', vol='EGARCH', p=5, q=5)
eth_egarch55_fit = eth_egarch55.fit()

eth_garch11 = arch_model(eth_return, mean='Constant', vol='GARCH', p=1, q=1)
eth_garch11_fit = eth_garch11.fit()
eth_egarch11 = arch_model(eth_return, mean='Constant', vol='EGARCH', p=1, q=1)
eth_egarch11_fit = eth_egarch11.fit()

usdt_garch33 = arch_model(usdt_return, mean='Constant', vol='GARCH', p=3, q=3)
usdt_garch33_fit = usdt_garch33.fit()
usdt_egarch33 = arch_model(usdt_return, mean='Constant', vol='EGARCH', p=3, q=3)
usdt_egarch33_fit = usdt_egarch33.fit()

usdt_garch11 = arch_model(usdt_return, mean='Constant', vol='GARCH', p=1, q=1)
usdt_garch11_fit = usdt_garch11.fit()
usdt_egarch = arch_model(usdt_return, mean='Constant', vol='EGARCH', p=3, q=2) # najit vhodne parametry
usdt_egarch_fit = usdt_egarch.fit()

print(btc_egarch11_fit)

forecast = btc_garch44_fit.forecast(start=0)

# LaTeX tabulka
data = []
data.append(['BTC', 'GARCH', *btc_garch44_fit.pvalues.values])
data.append(['BTC', 'EGARCH', *btc_egarch44_fit.pvalues.values])
data.append(['ETH', 'GARCH', *eth_garch55_fit.pvalues.values])
data.append(['ETH', 'EGARCH', *eth_egarch55_fit.pvalues.values])
data.append(['USDT', 'GARCH', *usdt_garch33_fit.pvalues.values])
data.append(['USDT', 'EGARCH', *usdt_egarch33_fit.pvalues.values])

df = pd.DataFrame(data, columns=['Asset', 'Model', 'mu', 'omega', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'beta1', 'beta2', 'beta3', 'beta4', 'beta5',])
df = df.set_index(['Asset', 'Model']).transpose()

latex_table = df.to_latex(index=True)
print(latex_table)

""" # Initialize an empty list to store the results
results = []

# Lists of cryptocurrencies and models
cryptos = ['BTC', 'ETH', 'USDt']
models = ['ARCH(1)', 'GARCH(1,1)', 'EGARCH(1,1)']

# Initialize an empty list to store the results
results = []

# Initialize an empty list to store the results
results = []

for asset_returns, asset_name in zip([btc_return, eth_return, usdt_return], ['BTC', 'ETH', 'USDt']):
    print(f"*************************************{asset_name}*************************************")
    
    # ARCH(1) Model
    arch_model_1 = arch_model(asset_returns, mean='Constant', vol='ARCH', p=1)
    arch_result_1 = arch_model_1.fit()
    
    # GARCH(1,1) Model
    garch_model_11 = arch_model(asset_returns, mean='Constant', vol='GARCH', p=1, q=1)
    garch_result_11 = garch_model_11.fit()
    
    # EGARCH(1,1) Model
    egarch_model_11 = arch_model(asset_returns, mean='Constant', vol='EGARCH', p=1, q=1)
    egarch_result_11 = egarch_model_11.fit()

    # Append the results to the list
    results.append({
        'Asset': asset_name,
        'Model': 'ARCH(1)',
        'Parameter P-values': arch_result_1.pvalues,
        'Loglikelihood': arch_result_1.loglikelihood,
        'AIC': arch_result_1.aic,
        'BIC': arch_result_1.bic
    })
    results.append({
        'Asset': asset_name,
        'Model': 'GARCH(1,1)',
        'Parameter P-values': garch_result_11.pvalues,
        'Loglikelihood': garch_result_11.loglikelihood,
        'AIC': garch_result_11.aic,
        'BIC': garch_result_11.bic
    })
    results.append({
        'Asset': asset_name,
        'Model': 'EGARCH(1,1)',
        'Parameter P-values': egarch_result_11.pvalues,
        'Loglikelihood': egarch_result_11.loglikelihood,
        'AIC': egarch_result_11.aic,
        'BIC': egarch_result_11.bic
    }) """

""" # Create a DataFrame from the results
df = pd.DataFrame(results)

# Set the index to Asset and Model
df = df.set_index(['Asset', 'Model'])

# Output the results as a LaTeX table
print(df.to_latex(escape=False)) """

### Plots ###

# BTC/ETH log(price)
""" plt.plot(btc_date[:2921], btc_open_log[:2921], label='BTC')
plt.plot(eth_date[:], eth_open_log[:], label='ETH')
plt.xlabel('Datum')
plt.ylabel('log(Cena)')
plt.title('Porovnání ceny BTC a ETH')
plt.legend()
plt.grid(True)
plt.show() """

# PACF
""" fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
plot_pacf(btc_return**2, ax=ax[0])
plot_pacf(eth_return**2, ax=ax[1])
plot_pacf(usdt_return**2, ax=ax[2])
ax[0].set_title('BTC')
ax[1].set_title('ETH')
ax[2].set_title('USDt')
ax[0].set_xlabel('Lag')
ax[1].set_xlabel('Lag')
ax[2].set_xlabel('Lag')
ax[0].set_ylabel('PACF')
ax[1].set_ylabel('PACF')
ax[2].set_ylabel('PACF')
plt.show() """

# BTC predikce volatility
""" predictions = forecast.variance['h.1']

prediction_dates = btc_date[:]

plt.figure(figsize=(10, 6))
plt.plot(btc_date[1:1460], btc_return[1:1460], label='Výnosy')
plt.plot(prediction_dates[1:1460], predictions[1:1460], color='orange', label='Predikce Volatility')
plt.title('Výnosy a predikce volatility BTC')
plt.legend()
plt.show() """





# In-sample fitting
""" 
PyFlux
maximalizovat věrohodnost
kaikeho kriterium
 """
# Out-of-sample forecasting
""" 
diebolt mariano test (porovnat modely)
arch-lm test (dostačující model?)
 """
# Forecast future values using the fitted model
""" 
log výnosy
 """
