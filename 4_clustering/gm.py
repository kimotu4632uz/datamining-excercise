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
results = model.predict(inputs)

print("Original  : {0}".format(labels.astype(np.int64)))
print("Clustering: {0}".format(results))

x_min = inputs[:, 0].min() - 1
x_max = inputs[:, 0].max() + 1
y_min = inputs[:, 1].min() - 1
y_max = inputs[:, 1].max() + 1
grid_interval = 0.02
x_grids, y_grids = np.meshgrid(
    np.arange(x_min, x_max, grid_interval),
    np.arange(y_min, y_max, grid_interval))

Fcand = [0, 1]
mycolors = ['r', 'b', 'g']
for i, mycolor in enumerate(mycolors):
    plt.scatter(inputs[results==i, 0], inputs[results==i, 1], color=mycolor, alpha=0.5) 

    means = [model.means_[i,j] for j in Fcand]
    cov = [[model.covariances_[i,j,k] for j in Fcand] for k in Fcand]

    rv = multivariate_normal(means, cov);
    z_grids = rv.pdf(np.dstack((x_grids, y_grids)))
    plt.contour(x_grids, y_grids, z_grids, levels=5, colors=mycolor)

plt.show();
