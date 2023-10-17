##model selection using AIC
data=newdf['Value']
def find_best_arima_order(data):
    p_values=range(0, 5)
    d_values=range(0, 3)
    q_values=range(0, 5)
    best_aic=float('inf')
    best_order=None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = sm.tsa.ARIMA(data, order=order)
                    results = model.fit()
                    aic = results.aic
                    if aic<best_aic:
                        best_aic = aic
                        best_order = order
                except:
                    continue
    return best_order
best_order=find_best_arima_order(data)
print("best order for ARIMA:",best_order)

#train
model=ARIMA(newdf['Value'],order=(2,0,1))
model_fit=model.fit()

#get the fitted values
arima_fitted_values=model_fit.fittedvalues

#model evaluation
print("arimaMSE=",round(MSE(newdf['Value'],arima_fitted_values),4))
print("arimaMADE=",round(MADE(newdf['Value'],arima_fitted_values),4))

#performing sequence forecasting
forecast=model_fit.get_forecast(steps=60)
arima_predicted_values=forecast.predicted_mean

#residual analysis
print(model_fit.summary())
residuals = model_fit.resid

# plot residual sequence graph
plt.style.use(['science','ieee'])
plt.grid (False)
plt.plot(residuals,linewidth=0.5)
plt.title('Residuals of ARIMA Model',fontsize=8)
plt.ylabel('Residuals',fontsize=6)
plt.xticks(range(0, 905, 180),rotation=0,fontsize=5)
plt.yticks(fontsize=5)
plt.savefig('residual.png', transparent=True,dpi=1000)
plt.show()
plt.clf()

# plot residual Q-Q plot
plt.style.use(['science','ieee'])
sm.qqplot(residuals, line='s',linewidth=2.0,marker='o', markersize=0.8)
plt.grid (False)
plt.title('Q-Q Plot of Residuals',fontsize=8)
plt.yticks(fontsize=5)
plt.xticks(fontsize=5)
plt.ylabel('Sample Quantiles',fontsize=6)
plt.xlabel('Theoretical Quantiles',fontsize=6)
plt.savefig('residualQQ.png', transparent=True,dpi=1000)
plt.show()
plt.clf()

#normality
statistic, p_value = stats.shapiro(residuals)
print("Kolmogorov-Smirnov Test:")
print("Statistic:", statistic)
print("p-value:", p_value)
#white noise test for residual sequence
rrresult_ljungbox=sm.stats.acorr_ljungbox(residuals,lags=10)  # specify lag order
print('result for Ljung-Box test：')
print('statistic：',rrresult_ljungbox)

#Visualization
plt.style.use(['science','ieee'])
plt.grid (False)
plt.plot(newdf['Date'],newdf['Value'], label='Original Data')
plt.plot(arima_fitted_values, label='Fitted Values')
plt.plot(arima_predicted_values, label='Predicted Values',linestyle='-')
plt.xticks(range(0, 905, 180),rotation=0,fontsize=5)
plt.yticks(fontsize=5)
plt.ylabel('Unemployment Rate',fontsize=6)
plt.legend(fontsize='small')
plt.title('Results for ARIMA Model',fontsize=8)
plt.savefig('ARIMA.png', transparent=True,dpi=1000)
plt.show()


