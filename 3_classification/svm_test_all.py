import joblib
import numpy as np

with open("iris_test.csv", 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',', usecols=(0,1,2,3,4))

inputs = data[:,0:4]
labels = data[:,4]

type0 = inputs[labels==0]
type1 = inputs[labels==1]

test_inputs = np.r_[type0, type1]
test_labels = np.r_[np.zeros(len(type0)),np.ones(len(type1))]

clf = joblib.load('svm_all.pkl')
results = clf.predict(test_inputs)

print("Answer : {0}".format(test_labels))
print("Predict: {0}".format(results))

