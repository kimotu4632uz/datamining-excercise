import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

with open("boston.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')

inputs = data[:,:13]
# inputs = inputs[:,np.newaxis] # convert [*,*,*,..] -> [[*],[*],[*],...]

outputs = data[:,13]

scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

lasso = Lasso().fit(inputs, outputs)

print('Coefficients: \n', lasso.coef_)
print('R2_score: \n', lasso.score(inputs, outputs))

# plot
# x_min = np.min(inputs)
# x_max = np.max(inputs)
# plot_x = np.arange(x_min,x_max,0.1)
# plot_x = plot_x[:,np.newaxis] # convert [*,*,*,..] -> [[*],[*],[*],...]
# plt.scatter(inputs[:,0], outputs,  color='black')
# plt.plot(plot_x[:,0], regr.predict(plot_x), color='red')
# plt.show()
