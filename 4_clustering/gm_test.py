from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

with open("iris.csv", 'r') as file:
    header = file.readline()
    data = np.loadtxt(file, delimiter=',', usecols=(0,1,2,3,4))

inputs = data[:,0:4]
labels = data[:,4]

K = 3
model = GaussianMixture(n_components=K).fit(inputs)

with open("iris_with_outlier.csv", 'r') as file:
    header = file.readline()
    inputs = np.loadtxt(file, delimiter=',', usecols=(0,1,2,3))

results = model.predict(inputs)
score = model.score_samples(inputs)
print("log score of iris_with_outlier: {0}".format(score))
print("min score: ", score.min())

Fx = 0
Fy = 3
Fcand = [Fx, Fy]
mycolors = ['r', 'b', 'g']

x_min = inputs[:, Fx].min() - 1
x_max = inputs[:, Fx].max() + 1
y_min = inputs[:, Fy].min() - 1
y_max = inputs[:, Fy].max() + 1
grid_interval = 0.02
x_grids, y_grids = np.meshgrid(
    np.arange(x_min, x_max, grid_interval),
    np.arange(y_min, y_max, grid_interval))

for i, mycolor in enumerate(mycolors):
    plt.scatter(inputs[results==i, Fx], inputs[results==i, Fy], color=mycolor, alpha=0.5) 

    means = [model.means_[i,j] for j in Fcand]
    cov = [[model.covariances_[i,j,k] for j in Fcand] for k in Fcand]

    rv = multivariate_normal(means, cov);
    z_grids = rv.pdf(np.dstack((x_grids, y_grids)))
    plt.contour(x_grids, y_grids, z_grids, levels=5, colors=mycolor)

min = score.argmin()
plt.scatter(inputs[min, Fx], inputs[min, Fy], color='k')
plt.show();
