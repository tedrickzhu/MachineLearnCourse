#encoding=utf-8

import numpy as np  
import matplotlib.pyplot as plt 


data = np.random.randint(0,10,(20,2))
data = data[data[:,0].argsort()]
plt.plot(data[:,0],data[:,1])

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
#设置标题  
ax1.set_title('Scatter Plot')  
#设置X轴标签  
plt.xlabel('X')  
#设置Y轴标签  
plt.ylabel('Y')  
#画散点图  
ax1.scatter(x=data[:,0],y=data[:,1],c = 'r',marker = 'o')  
#设置图标  
plt.legend('x1')  
#显示所画的图  
plt.show() 
