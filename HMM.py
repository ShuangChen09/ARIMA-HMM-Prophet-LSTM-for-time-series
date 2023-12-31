##prepare data
data=pd.read_excel(r"C:\Users\LX\Desktop\dataset.xlsx").iloc[:, 1:3]['Value']
train_data=data[:905]
train_data=np.array(train_data).reshape(-1, 1)
#establish HMM
model=hmm.GaussianHMM(n_components=10,n_iter=100)
#fit the model
model.fit(train_data)
#Forecasting the next 60 time steps
predicted_states, _=model.sample(n_samples=60)
predicted_values=predicted_states.flatten()
# visualization
plt.style.use(['science','ieee'])
plt.grid (False)
plt.plot(newdf['Date'],newdf['Value'], label='Original Data')
hidden_states=model.predict(train_data)
hmm_fitted_values=model.means_[hidden_states].flatten()
plt.plot(hmm_fitted_values, label='Fitted Values')
start_value=train_data[-1]
predicted_values=model.sample(n_samples=60, random_state=0)[0].flatten()
hmm_predicted_values=np.concatenate([start_value, predicted_values])
plt.plot(range(len(data),len(data)+len(hmm_predicted_values)), hmm_predicted_values, label='Predicted Values',linestyle='-')
plt.xticks(range(0, 905, 180),rotation=0,fontsize=5)
plt.yticks(fontsize=5)
plt.ylabel('Unemployment Rate',fontsize=6)
plt.legend(fontsize='small')
plt.title('Results for HMM Model',fontsize=8)
plt.savefig('HMM.png', transparent=True,dpi=1000)
plt.show()
plt.clf()
print("hmmMSE=",round(MSE(newdf['Value'],hmm_fitted_values),4))
print("hmmMADE=",round(MADE(newdf['Value'],hmm_fitted_values),4))
