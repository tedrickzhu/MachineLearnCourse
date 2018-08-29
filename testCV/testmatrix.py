import numpy as np

###矩阵a
a = np.floor(10 * np.random.rand(2, 2))
###a
# array([[8., 5.],
#        [1., 6.]])

###矩阵b
b = np.floor(10 * np.random.rand(2, 2))
# array([[1., 9.],
#        [8., 5.]])
# 我们随机生成了a，b这两个矩阵，下面进行合并操作：

###hstack()在行上合并
np.hstack((a, b))
# array([[8., 5., 1., 9.],
#        [1., 6., 8., 5.]])

####vstack()在列上合并
np.vstack((a, b))
# array([[8., 5.],
#        [1., 6.],
#        [1., 9.],
#        [8., 5.]])