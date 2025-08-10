# ML-using-Python-notes
Day-1(06/08/25):
#1 importing required modules
!pip install numpy
import numpy as np
#create nd-array from a list
data1 =[11, 22,33,44.0,55,66,77,88,'AEC',True,False]
print(data1,len(data1),type(data1))
arr1 =np.array(data1)

print(arr1, len(arr1),type(arr1), arr1.dtype)
o/p-Successfully installed numpy-2.3.2
[11, 22, 33, 44.0, 55, 66, 77, 88, 'AEC', True, False] 11 <class 'list'>
['11' '22' '33' '44.0' '55' '66' '77' '88' 'AEC' 'True' 'False'] 11 <class 'numpy.ndarray'> <U32
#2
data1 = range(1,20,3) #list containing 0 to 9
print(data1)
arr1 = np.array(data1)
print(arr1)
o/p-range(1, 20, 3)
[ 1  4  7 10 13 16 19]
#3
data2 =[range(1,5),range(6,10)]
print(data2)
arr2 = np.array(data2)
print(arr2)#creating 2-D array
for item in np.nditer(arr2):
    print(item, end=",")
o/p-[range(1, 5), range(6, 10)]
[[1 2 3 4]
 [6 7 8 9]]
1,2,3,4,6,7,8,9,

#4
data1 = [11,22,33,44]
arr1=np.array(data1)
print(data1,arr1)
result=list(arr1)
print(arr1,result)
result= arr1.tolist()
print(arr1,result)
o/p
[11, 22, 33, 44] [11 22 33 44]
[11 22 33 44] [np.int64(11), np.int64(22), np.int64(33), np.int64(44)]
[11 22 33 44] [11, 22, 33, 44]
#5
data2 =[[11,22,33,44],[55,66,77,88]]
arr2 =np.array(data2)
print(data2)
print(arr2)
result =list(arr2)
print(arr2)
print(result)
result= arr2.tolist()
print(arr2)
print(result)
o/p-[[11, 22, 33, 44], [55, 66, 77, 88]]
[[11 22 33 44]
 [55 66 77 88]]
[[11 22 33 44]
 [55 66 77 88]]
[array([11, 22, 33, 44]), array([55, 66, 77, 88])]
[[11 22 33 44]
 [55 66 77 88]]
[[11, 22, 33, 44], [55, 66, 77, 88]]
#6
#examine arrays
arr1 = np.array([11,22,33,44,55,66,77,88])
print(arr1, type(arr1),id(arr1))
print(arr1.dtype, arr1.shape,len(arr1),arr1.ndim)
print(arr1.dtype,arr1.shape,len(arr1),arr1.ndim)
print(arr1.size)
o/p-[11 22 33 44 55 66 77 88] <class 'numpy.ndarray'> 1808502471536
int64 (8,) 8 1
int64 (8,) 8 1
8
#7
var1=(8)
print(var1, type(var1))
var1 =(8,)#singleton representation of a tuple
print(var1,type(var1))
o/p=8 <class 'int'>
(8,) <class 'tuple'>
#8
arr2 = np.array([[11,22,33,44],[55,66,77,88]])
print(arr2,type(arr2),id(arr2))
print(arr2.size)
o/p-[[11 22 33 44]
 [55 66 77 88]] <class 'numpy.ndarray'> 1808502467888
8
#9
#create special arrays
print(np.zeros(10))
print(np.zeros(10,np.int32))
print(np.zeros(10).astype(int))
print(np.zeros((2,5)))
print(np.zeros((2,5),np.int32))
print(np.zeros((2,5)).astype(int))
o/p-[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 0 0 0 0 0 0]
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
[[0 0 0 0 0]
 [0 0 0 0 0]]
[[0 0 0 0 0]
 [0 0 0 0 0]]
 #10
 print(np.zeros((3,5))+5) #vector and scalar operation
print(np.ones((3,5)) *5)
o/p-[[5. 5. 5. 5. 5.]
 [5. 5. 5. 5. 5.]
 [5. 5. 5. 5. 5.]]
[[5. 5. 5. 5. 5.]
 [5. 5. 5. 5. 5.]
 [5. 5. 5. 5. 5.]]
 #11
print(np.linspace(1,10,5))
print(np.linspace(1,50,11))
o/p-[ 1.    3.25  5.5   7.75 10.  ]
[ 1.   5.9 10.8 15.7 20.6 25.5 30.4 35.3 40.2 45.1 50. ]
#12
print(np.logspace(0,3,4))
print(np.logspace(0,3,4,base=2))
print(np.logspace(0,8,9,base = np.e))
o/p-[   1.   10.  100. 1000.]
[1. 2. 4. 8.]
[1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03
 2.98095799e+03]
 #13
 arr1 = np.arange(1,11,2)
print(arr1,type(arr1))
arr1 = np.arange(11,0,-2)
print(arr1,type(arr1))
arr1 = np.arange(1,11,2).astype(float)
print(arr1,type(arr1))
o/p-[1 3 5 7 9] <class 'numpy.ndarray'>
[11  9  7  5  3  1] <class 'numpy.ndarray'>
[1. 3. 5. 7. 9.] <class 'numpy.ndarray'>
#14
#reshape of ndarray
print(np.arange(20))
print(np.arange(20).reshape(4,5))
print(np.arange(20).reshape(4,-1))
print(np.arange(20).reshape(-1,5))
o/p-[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
 #15
 print(np.arange(20).reshape(2,-1,5))
 o/p-[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]]
  #16
  arr2=np.arange(20).reshape(2,10)
print(arr2)
arr1=arr2.flatten()
print(arr1)
o/p-
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]]
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
#17
arr2 = np.arange(20).reshape(5,4)
print(arr2)
arr_transpose= arr2.T
print(arr_transpose)
arr_transpose = arr2.transpose()
print(arr_transpose)
o/p-[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]]
[[ 0  4  8 12 16]
 [ 1  5  9 13 17]
 [ 2  6 10 14 18]
 [ 3  7 11 15 19]]
[[ 0  4  8 12 16]
 [ 1  5  9 13 17]
 [ 2  6 10 14 18]
 [ 3  7 11 15 19]]
 #18
print(np.linspace(1,10,5))
print(np.linspace(1,50,11))
o/p-[ 1.    3.25  5.5   7.75 10.  ]
[ 1.   5.9 10.8 15.7 20.6 25.5 30.4 35.3 40.2 45.1 50. ]
#19
arr2 = np.arange(20).reshape(5, 4)
print(arr2)
arr_transpose = arr2.T
print(arr_transpose)
arr_transpose = arr2.transpose()
print(arr_transpose)
o/p-[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]]
[[ 0  4  8 12 16]
 [ 1  5  9 13 17]
 [ 2  6 10 14 18]
 [ 3  7 11 15 19]]
[[ 0  4  8 12 16]
 [ 1  5  9 13 17]
 [ 2  6 10 14 18]
 [ 3  7 11 15 19]]
 #20
 matrix = np.arange(10,dtype=float).reshape((2,5))
print(matrix)
print("stored in the C-tyle order(Row major):")
cmatrix = matrix.copy(order='C')
print(cmatrix)
for item in np.nditer(cmatrix):
    print(item, end=",")    
o/p-[[0. 1. 2. 3. 4.]
 [5. 6. 7. 8. 9.]]
stored in the C-tyle order(Row major):
[[0. 1. 2. 3. 4.]
 [5. 6. 7. 8. 9.]]
0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
#21
matrix = np.arange(10,dtype=float).reshape((2,5))
print(matrix)
print("stored in the F-style order(Column major):")
fmatrix = matrix.copy(order='F')
print(fmatrix)
for item in np.nditer(fmatrix):
    print(item, end=",")    
[[0. 1. 2. 3. 4.]
 [5. 6. 7. 8. 9.]]
stored in the F-style order(Column major):
[[0. 1. 2. 3. 4.]
 [5. 6. 7. 8. 9.]]
0.0,5.0,1.0,6.0,2.0,7.0,3.0,8.0,4.0,9.0,
#22
matrix = np.arange(10, dtype=np.int8)
print(matrix,len(matrix))
matrix = np.append(matrix,[11,22,33,44])
print(matrix,len(matrix))
o/p-[0 1 2 3 4 5 6 7 8 9] 10
[ 0  1  2  3  4  5  6  7  8  9 11 22 33 44] 14
#23 Tuple
matrix = np.arange(10, dtype=np.int8)
print(matrix,len(matrix))
matrix = np.append(matrix,[11,22,33,44])
print(matrix,len(matrix))
matrix = np.append(matrix, [55, 66, 77, 88])
print(matrix, len(matrix))
o/p-[0 1 2 3 4 5 6 7 8 9] 10
[ 0  1  2  3  4  5  6  7  8  9 11 22 33 44] 14
[ 0  1  2  3  4  5  6  7  8  9 11 22 33 44 55 66 77 88] 18
#24
matrix = np.arange(10, dtype=np.int8)
print(matrix,len(matrix))
matrix = np.append(matrix,[11,22,33,44])
print(matrix,len(matrix))
matrix = np.append(matrix, [55, 66, 77, 88])
print(matrix, len(matrix))
matrix = np.delete(matrix,[5,6,8])
print(matrix)
o/p-[0 1 2 3 4 5 6 7 8 9] 10
[ 0  1  2  3  4  5  6  7  8  9 11 22 33 44] 14
[ 0  1  2  3  4  5  6  7  8  9 11 22 33 44 55 66 77 88] 18
[ 0  1  2  3  4  7  9 11 22 33 44 55 66 77 88]
#25
matrix = np.array([22,44,77,11,88,66])
print (matrix)
matrix1=np.sort(matrix)#ascending order
print(matrix1)
matrix1 = -np.sort(-matrix)#descending order
print(matrix1)
o/p-[22 44 77 11 88 66]
[11 22 44 66 77 88]
[88 77 66 44 22 11]
#26
matrix1=np.array([[11,22],[33,44]])
print(matrix1)
matrix2=np.array([[55,66],[77,88]])
print((matrix2))
#joining these two matrices along axis =0
matrix3 = np.concatenate((matrix1,matrix2))
print(matrix3)
o/p-[[11 22]
 [33 44]]
[[55 66]
 [77 88]]
[[11 22]
 [33 44]
 [55 66]
 [77 88]]
 #27
 matrix1=np.array([[11,22],[33,44]])
print(matrix1)
matrix2=np.array([[55,66],[77,88]])
print((matrix2))
#joining these two matrices along axis =0
matrix3 = np.concatenate((matrix1,matrix2))
print(matrix3)
#joining these two matrices along axis =1
matrix4 = np.concatenate((matrix1,matrix2),axis=1)
print(matrix4,matrix4.shape,matrix4.ndim)
o/p-[[11 22]
 [33 44]]
[[55 66]
 [77 88]]
[[11 22]
 [33 44]
 [55 66]
 [77 88]]
[[11 22 55 66]
 [33 44 77 88]] (2, 4) 2
 #28
 print("Stack the two arrays along axis 0")
matrix3 =np.stack((matrix1,matrix2))
print(matrix3)
matrix4= np.stack((matrix1,matrix2), axis=1)
print(matrix4,matrix4.shape,matrix4.ndim)

o/p-Stack the two arrays along axis 0
[[[11 22]
  [33 44]]

 [[55 66]
  [77 88]]]
[[[11 22]
  [55 66]]

 [[33 44]
  [77 88]]] (2, 2, 2) 3
#29
print("Stack the two arrays along axis 0")
matrix3 = np.stack((matrix1, matrix2))
print(matrix3, matrix3.shape, matrix3.ndim)
matrix4 = np.stack((matrix1, matrix2), axis = 1)
print(matrix4, matrix4.shape, matrix4.ndim)
o/p-Stack the two arrays along axis 0
[[[11 22]
  [33 44]]

 [[55 66]
  [77 88]]] (2, 2, 2) 3
[[[11 22]
  [55 66]]

 [[33 44]
  [77 88]]] (2, 2, 2) 3
  #30
  arr2=np.arange(20).reshape(4,5)
print(arr2)
print(arr2[2,:])
print(arr2[:,2])
print(arr2[1:3,2:4])
o/p-[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
[10 11 12 13 14]
[ 2  7 12 17]
[[ 7  8]
 [12 13]]
 #31
arr2=np.arange(20).reshape(4,5)
print(arr2)
print(arr2[2,:])
print(arr2[:,2])
print(arr2[1:3,2:4])
print(arr2[1:,2:])
print(arr2[0::2])
print(arr2[0::2,1::2])
o/p-[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
[10 11 12 13 14]
[ 2  7 12 17]
[[ 7  8]
 [12 13]]
[[ 7  8  9]
 [12 13 14]
 [17 18 19]]
[[ 0  1  2  3  4]
 [10 11 12 13 14]]
[[ 1  3]
 [11 13]]
 #32
 arr1 = np.arange(10)
print(arr1,type(arr1))
arr_copy=arr1[2:5].copy() #creating a copy
print(arr_copy)
arr_copy[:]=100
print(arr1)
print(arr_copy)
o/p-[0 1 2 3 4 5 6 7 8 9] <class 'numpy.ndarray'>
[2 3 4]
[0 1 2 3 4 5 6 7 8 9]
[100 100 100] 
#33
import numpy as np

arr = np.zeros((6, 6), dtype=int)
arr[0, :] = 1
arr[-1, :] = 1
arr[:, 0] = 1
arr[:, -1] = 1
print(arr)
o/p-[[1 1 1 1 1 1]
 [1 0 0 0 0 1]
 [1 0 0 0 0 1]
 [1 0 0 0 0 1]
 [1 0 0 0 0 1]
 [1 1 1 1 1 1]]
 or
 arr2=np.zeros((6,6))
arr2[[0,-1],:] = 1
arr2[:,[0,-1]] = 1
print(arr2)
 #34
arr1 = np.array([100, 20, -33, 44, 90, 35, -22])
print(arr1)
print(arr1<0)
print(arr1[arr1 < 0])
print(~(arr1 < 0))
print(arr1[~(arr1 < 0)])
o/p-[100  20 -33  44  90  35 -22]
[False False  True False False False  True]
[-33 -22]
[ True  True False  True  True  True False]
[100  20  44  90  35]
#35
arr1 = np.array([100, 20, -33, 44, 90, 35, -22])
print(arr1)
print(arr1%2==True)
print(arr1[arr1%2==0])#retrieving even numbers
print(arr1[~(arr1%2==0)])#retrieving odd numbers
o/p-[100  20 -33  44  90  35 -22]
[False False  True False False  True False]
[100  20  44  90 -22]
[-33  35]
#36
arr1 = np.array([100,20,-33,44,90,35,-22])
print(arr1)
print((arr1>=20) & (arr1 <=50))
print(arr1[(arr1 >=20) & (arr1 <=50)])
print(np.__version__)
print(arr1.nbytes) #retrieving numbers between 20 and 50
o/p-[100  20 -33  44  90  35 -22]
[False  True False  True False  True False]
[20 44 35]
2.3.2
56
#37
arr1 = np.array(['bob','pat','tom','bob','jim','john'])
print(arr1=='BOB')
print(arr=='bob')
print(arr1[arr1!='bob'])
arr1[arr1=='bob'] = 'tim'
print(arr1) #changing values
o/p-[False False False False False False]
[[False False False False False False]
 [False False False False False False]
 [False False False False False False]
 [False False False False False False]
 [False False False False False False]
 [False False False False False False]]
['pat' 'tom' 'jim' 'john']
['tim' 'pat' 'tom' 'tim' 'jim' 'john']
#38
nums=np.arange(6)
nums_sqrt = np.sqrt(nums)
print(nums_sqrt)
print(np.round(nums_sqrt,1),np.round(nums_sqrt,4))
print(np.ceil(nums_sqrt))
print(np.floor(nums_sqrt))
o/p-[0.         1.         1.41421356 1.73205081 2.         2.23606798]
[0.  1.  1.4 1.7 2.  2.2] [0.     1.     1.4142 1.7321 2.     2.2361]
[0. 1. 2. 2. 2. 3.]
[0. 1. 1. 1. 2. 2.]
#39
#dealing with random numbers
nums = np.random.randn(10)
print(nums)
print(np.average(nums),np.std(nums))
o/p-[ 0.51421423 -0.9222735   0.10628887 -0.96270085  0.75370239  1.24193707
 -1.56016315  0.42612416  1.13627468  0.18018588]
0.09135897871684981 0.8955100910703041
#40
nums = np.random.randn(500000)
print(nums[:10])
print(np.average(nums),np.std(nums))
o/p-[ 1.58082713  1.04704363 -0.43345807  0.14128454  0.28660876  1.12530517
 -0.35419236  1.56857163 -1.4136639   0.69523389]
-0.002836505603716415 0.999861747073846 
#41
np.random.seed(10)
print(np.random.randn(10))
o/p-[ 1.3315865   0.71527897 -1.54540029 -0.00838385  0.62133597 -0.72008556
  0.26551159  0.10854853  0.00429143 -0.17460021]
  #42
  print(np.random.randint(0,5,20))
  o/p-[3 2 1 0 4 1 3 3 1 4 1 4 1 1 4 3 2 0 3 4]
  #43
  pt1=np.random.randn(10)
pt2=np.random.randn(10)
e_dist =0
for i in range(len(pt1)):
    e_dist += (pt1[i]-pt2[i])**2
    print(e_dist **0.5)
    print (np.sqrt(np.sum((pt1-pt2)**2)))
    print(np.sqrt(np.sum(pow(np.subtract(pt1, pt2),2))))
o/p-1.9403940858656452
3.395792881516465
3.395792881516465
2.384539110887282
3.395792881516465
DAY_2
Data Gathering
Data Cleaning/Santization
Exploring the data analytics
#45
import numpy as np  
import pandas as pd
 # creating Series from a list
list_data = [101, 202, 404, 303, 505]
print(list_data, type(list_data))
s = pd.Series(data=list_data, index=['red','green','brown','blue','magenta'])
print(s)
print(s[0], s['red'], s[4], s['magenta'])  # indexing
print(s[-1], s['magenta'])  # indexing
print(s[2:])  # slicing (2nd inde onwards)
print(s[:-1])  # slicing (from first index till the last but one)
o/p-[101, 202, 404, 303, 505] <class 'list'>
red        101
green      202
brown      404
blue       303
magenta    505
dtype: int64
101 101 505 505
505 505
brown      404
blue       303
magenta    505
dtype: int64
red      101
green    202
brown    404
blue     303
dtype: int64
#46
print(s)
print(s.sort_index())
print(s.sort_values())
print(s.argmin(), s.argmax(), len(s), s.count(), s.mean(), s.var(), s.std(), s.max(), s.min(), s.sum())
o/p-red        101
green      202
brown      404
blue       303
magenta    505
dtype: int64
blue       303
brown      404
green      202
magenta    505
red        101
dtype: int64
red        101
green      202
blue       303
brown      404
magenta    505
dtype: int64
0 4 5 5 303.0 25502.5 159.69502183850315 505 101 1515
#47
user_data = [['alice',19,'F','student'],['john',26,'M','student']]
user_columns = ['name','age','gender','job']
user1 = pd.DataFrame(data=user_data, columns=user_columns)
user1
o/p-user_data = [['alice',19,'F','student'],['john',26,'M','student']]
user_columns = ['name','age','gender','job']
user1 = pd.DataFrame(data=user_data, columns=user_columns)
user1
#48
user_data = dict(name=['eric','paul'], age=[22,58], gender=['M','F'], job=['student','manager'])
print(user_data, type(user_data))
user2 = pd.DataFrame(data=user_data)
user2
o/p-
name	age	gender	job
0	eric	22	M	student
1	paul	58	F	manager
#49
user_data = {'name':['peter','paul'], 'age':[33,44], 'gender':['M','F'], 'job':['engineer','scientist']}
print(user_data, type(user_data))
user3 = pd.DataFrame(data=user_data)
user3
o/p-
name	age	gender	job
0	peter	33	M	engineer
1	paul	44	F	scientist
#50
users = pd.concat([user1, user2, user3], ignore_index=True)
print(users)
o/p-    name  age gender        job
0  alice   19      F    student
1   john   26      M    student
2   eric   22      M    student
3   paul   58      F    manager
4  peter   33      M   engineer
5   paul   44      F  scientist
#51
arr_data=np.array(users)
print(arr_data, type(arr_data))
arr_data=users.to_numpy()
print(arr_data, type(arr_data))
o/p-[['alice' 19 'F' 'student']
 ['john' 26 'M' 'student']
 ['eric' 22 'M' 'student']
 ['paul' 58 'F' 'manager']
 ['peter' 33 'M' 'engineer']
 ['paul' 44 'F' 'scientist']] <class 'numpy.ndarray'>
[['alice' 19 'F' 'student']
 ['john' 26 'M' 'student']
 ['eric' 22 'M' 'student']
 ['paul' 58 'F' 'manager']
 ['peter' 33 'M' 'engineer']
 ['paul' 44 'F' 'scientist']] <class 'numpy.ndarray'>
 #52
 #join dataframes
dict_data=dict(name=['alice','john','eric','paul','peter'], hieght=[165,180,175,180,181])
user5=pd.DataFrame(data=dict_data)
user5
o/p-
name	hieght
0	alice	165
1	john	180
2	eric	175
3	paul	180
4	peter	181
#53
shape-no.of rows,no. of columns in form of tuples
size-row * column
len-no of rows
users.shape[0]=rows
users.shape[1]=columns





























 













































