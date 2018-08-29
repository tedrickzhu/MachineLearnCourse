#encoding = utf-8
__author__='zzy'
__date__ ='2018-08-08'

from tensorflow.contrib.keras.api.keras.models import load_model
from cifar_data_load import getTestDataByLabel
import numpy as np

if __name__ == '__main__':
    # 载入训练好的模型
    model = load_model("lenet-no-activation-model.h5")
    # 获取测试集的数据
    X = getTestDataByLabel('data')
    Y = getTestDataByLabel('labels')
    # 统计预测正确的图片的数目
    probility = model.predict(X)
    #返回沿轴axis最大值的索引
    result = np.argmax(probility, 1)
    print(type(result),result.shape,result)
    #print(Y.shape,type(result),result.shape)
    print(np.sum(np.equal(Y, result)))