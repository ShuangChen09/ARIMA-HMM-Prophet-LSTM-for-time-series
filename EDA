#stationarity test
#unit root test
result=adfuller(newdf['Value'])
print('ADF：', result[0])
print('p：', result[1])
print('临界值：', result[4])
#plotting ACF (AutoCorrelation Function) graph
plt.style.use(['science','ieee','notebook'])
plot_acf(newdf['Value'], lags=20)
plt.ylim(-1.2,1.2)
plt.grid (False)
plt.savefig('ACF.png', transparent=True,dpi=1000)
plt.show()
#plotting PACF (Partial AutoCorrelation Function) graph
plot_pacf(newdf['Value'], lags=20)
plt.ylim(-1.2,1.2)
plt.grid (False)
plt.savefig('PACF.png', transparent=True,dpi=1000)
plt.show()

# Ljung-Box test
result_ljungbox=sm.stats.acorr_ljungbox(newdf['Value'],lags=10)  # specify lag order
print('Ljung-Box test result：')
print('statistics：',result_ljungbox)
