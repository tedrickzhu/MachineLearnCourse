#encoding=utf-8

#作者：yegle
#链接：https://www.zhihu.com/question/22735835/answer/22460886
#来源：知乎
#著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

import numpy as np

from functools import reduce
from operator import add

data = [[121.1, 0.02, 0.02],
        [121.1, 0.05, 0.05],
        [122.1, 0.56, 0.6],
        [122.1, 3.79, 4.04],
        [123.1, 93.75, 100.0],
        [123.1, 0.01, 0.01],
        [124.1, 0.01, 0.01],
        [124.1, 1.01, 1.08],
        [124.1, 0.11, 0.11],
        [124.1, 0.05, 0.06],
        [125.1, 0.39, 0.41]]

def reduce_callback(d, i):
    key = i[0]
    try:
        d[key] = list(map(add, d[key], i[1:]))
    except KeyError:
        d[key] = i[1:]
    return d

result = reduce(reduce_callback, data, {})
result_list = [[k]+v for k,v in result.items()]
print(result_list)


data = np.reshape(data,(11,3))
index_list =np.lexsort(-data[:,::-1].T)
print(index_list)

# python
# 按二维数组的某行或列排序(numpy
# lexsort)
# lexsort支持对数组按指定行或列的顺序排序；是间接排序，lexsort不修改原数组，返回索引。
# （对应lexsort
# 一维数组的是argsort
# a.argsort()
# 这么使用就可以；argsort也不修改原数组， 返回索引）
#
# 默认按最后一行元素有小到大排序, 返回最后一行元素排序后索引所在位置。
# 设数组a, 返回的索引ind，ind返回的是一维数组
# 对于一维数组, a[ind]
# 就是排序后的数组。
# 对于二维数组下面会详细举例。
#
# import numpy as np
#
# >> > a
# array([[2, 7, 4, 2],
#        [35, 9, 1, 5],
#        [22, 12, 3, 2]])
#
# 按最后一列顺序排序
# >> > a[np.lexsort(a.T)]
# array([[22, 12, 3, 2],
#        [2, 7, 4, 2],
#        [35, 9, 1, 5]])
#
# 按最后一列逆序排序
# >> > a[np.lexsort(-a.T)]
# array([[35, 9, 1, 5],
#        [2, 7, 4, 2],
#        [22, 12, 3, 2]])
#
# 按第一列顺序排序
# >> > a[np.lexsort(a[:, ::-1].T)]
# array([[2, 7, 4, 2],
#        [22, 12, 3, 2],
#        [35, 9, 1, 5]])
#
# 按最后一行顺序排序
# >> > a.T[np.lexsort(a)].T
# array([[2, 4, 7, 2],
#        [5, 1, 9, 35],
#        [2, 3, 12, 22]])
#
# 按第一行顺序排序
# >> > a.T[np.lexsort(a[::-1, :])].T
# array([[2, 2, 4, 7],
#        [5, 35, 1, 9],
#        [2, 22, 3, 12]])
# 转载请注明出处： http: // www.cnblogs.com / liyuxia713 /