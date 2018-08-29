#encoding=utf-8

import scipy.io as sio    
import matplotlib.pyplot as plt    
import numpy as np    
  
# 创建4个变量，并赋予相应的取值  
#sio.savemat('testpython.mat', {'a': 1,'b': 2,'c': 3,'d': 4})   
  
# 创建了一个变量x，并赋予一个矩阵  
#sio.savemat('testpython2.mat', {'x': [[1, 2, 3, 4],[ 5, 6, 7, 8]]})  
  
#data =  sio.loadmat('testpython.mat')  
#x1 = data['a']  
#x2 = data['b']  
#x3 = data['c']  
#x4 = data['d']  
  
#sio.whosmat('testpython.mat')  

matfile = sio.loadmat('ConcreteCompressiveStrength.mat')
print(matfile.keys())

data = matfile['z']

#print(type(data))
#print(data)
print(data.shape)
print(sum(data[:,8])/len(data))

