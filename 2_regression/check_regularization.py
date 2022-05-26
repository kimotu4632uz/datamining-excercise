import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

with open("out.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',')

inputs = data[:,0]
inputs = np.array([[x, x**2, x**3] for x in inputs])
#inputs = inputs[:,np.newaxis]
outputs = data[:,1]
regr = linear_model.LinearRegression()
regr.fit(inputs,outputs)
print('Coefficients: \n', regr.coef_)

plt.scatter(inputs[:,0], outputs,  color='black')
plot_x = np.arange(0,1,0.02)
plot_x_i = np.array([[x, x**2, x**3] for x in plot_x])
#plot_x = plot_x[:,np.newaxis]
plt.plot(plot_x, regr.predict(plot_x_i), color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
