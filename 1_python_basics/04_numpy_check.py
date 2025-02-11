import numpy as np

mylist = [1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9]
data = np.array(mylist)
mat = data.reshape(3,3)

print("**numpy.array形式の data**")
print(data)
print("**data.reshape(3,3) -> 3x3行列**")
print(mat)
print("**mat[0,1] -> (0,1)要素**")
print(mat[0,1])
print("**mat[0,:] -> 1次元目の1番目の要素すべて**")
print(mat[0,:])
print("**mat[:,1] -> 2次元目の2番目の要素すべて**")
print(mat[:,1])
print("**mat.T -> 転置**")
print(mat.T)

print('**2x3 cut**')
print(mat[:, 1:])

output = np.array([[10.1], [10.2], [10.3]])
mat_all = np.c_[mat, output]

print('**mat_all**')
print(mat_all)

print('**mat^-1**')
print(np.linalg.inv(mat))

print('**mat^-1 * mat**')
print(np.dot(np.linalg.inv(mat), mat))

data1 = np.array([[1.1,2.2],[3.3,4.4]])
data2 = np.array([[5.5,6.6],[7.7,8.8]])

print("**data1**")
print(data1)
print("**data2**")
print(data2)
print("**data1+data2 -> 足し算**")
print(data1+data2)
print("**data1*data1 -> 要素ごとのかけ算**")
print(data1*data2)
print("**np.dot(data1,data2) -> 行列としてのかけ算**")
print(np.dot(data1,data2))
print("**np.r_[data1,data2] -> １次元目でつなげる**")
print(np.r_[data1,data2])
print("**np.c_[data1,data2] -> ２次元目でつなげる**")
print(np.c_[data1,data2])
