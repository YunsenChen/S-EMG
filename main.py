# 读取arduino串口数据并保存
import serial
import pandas as pd
import time
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import math

# sets up serial connection (make sure baud rate is correct - matches Arduino)
# 设置串口号和波特率和Arduino匹配
ser = serial.Serial('com6', 115200)
# a为储存数据的列表
data_front = []
data_back = []
data_left = []
data_right = []

count = 0
print("前")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    # reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("front is null")

        else:
            data_front.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("后")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    # reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("back is null")

        else:
            data_back.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("左")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("left is null")

        else:
            data_left.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("右")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)

        if np.any(pd.DataFrame(data).isnull()):
            print("right is null")

        else:
            data_right.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("前")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)

        if np.any(pd.DataFrame(data).isnull()):
            print("front is null")

        else:
            data_front.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("后")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("back is null")

        else:
            data_back.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("左")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("left is null")
        else:
            data_left.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("右")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("right is null")

        else:
            data_right.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("前")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("front is null")

        else:
            data_front.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("后")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)

        if np.any(pd.DataFrame(data).isnull()):
            print("back is null")

        else:
            data_back.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("左")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("left is null")

        else:
            data_left.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

count = 0
print("右")
time.sleep(1)
while count != 2000:  # 30可以根据需要设置，while(True)：代表一直读下去
    data = ser.readline()  # 按行读取串口数据进来
    try:
        data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
        data = data[:-2]
        data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
        data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
        # print(data)
        if np.any(pd.DataFrame(data).isnull()):
            print("right is null")

        else:
            data_right.append(data)  # 添加到列表里
            # print(count)
            count += 1
    except:
        pass

df_front = pd.DataFrame(data_front)  # 转化为df格式数据
df_back = pd.DataFrame(data_back)
df_right = pd.DataFrame(data_right)
df_left = pd.DataFrame(data_left)
# print(df)
# df.to_excel('.\0.xls', header=False, index=False)
# ser.close()

E0_emg = np.array(df_front.iloc[:, :3])
E1_emg = np.array(df_back.iloc[:, :3])
E2_emg = np.array(df_left.iloc[:, :3])
E3_emg = np.array(df_right.iloc[:, :3])

E0_label = np.full(E0_emg.shape[0], 0)
E1_label = np.full(E1_emg.shape[0], 1)
E2_label = np.full(E2_emg.shape[0], 2)
E3_label = np.full(E3_emg.shape[0], 3)

emg = np.vstack((E0_emg, E1_emg, E2_emg, E3_emg))
label = np.vstack((E0_label, E1_label, E2_label, E3_label))
label = label.ravel()

# import matplotlib.pyplot as plt
# print(emg.shape)
# print(label.shape)
# print(label)
# plt.plot(emg/5000)
# plt.plot(label)
# plt.show()


classes = 4
timeWindow = 200
strideWindow = 200

imageData = []
imageLabel = []
imageLength = 200

for i in range(classes):
    index = [];
    for j in range(label.shape[0]):
        if (label[j] == i):
            index.append(j)

    iemg = emg[index, :]
    length = math.floor((iemg.shape[0] - imageLength) / imageLength)
    print("class ", i, " number of sample: ", iemg.shape[0], length)

    for j in range(length):
        subImage = iemg[imageLength * j:imageLength * (j + 1), :]
        imageData.append(subImage)
        imageLabel.append(i)

imageData = np.array(imageData)
# print(imageData.shape)
# print(len(imageLabel))


from feature_utils import *

featureData = []
featureLabel = []
# classes = 49
timeWindow = 200
strideWindow = 200
from tqdm import tqdm

for i in tqdm(range(imageData.shape[0])):
    # for i in tqdm(range(2)):
    rms = featureRMS(imageData[i])
    mav = featureMAV(imageData[i])
    wl = featureWL(imageData[i])
    zc = featureZC(imageData[i])
    ssc = featureSSC(imageData[i])
    var = featureVAR(imageData[i])
    wa = featureWA(imageData[i])
    me = featureME(imageData[i])
    mcv = featureMCV(imageData[i])
    mfp = featureMPF(imageData[i])
    ar0, ar1, ar2, ar3, ar4, ar5, ar6 = featureAR(imageData[i])
    sm2 = featureSM2(imageData[i])
    mf = featureMF(imageData[i])
    e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5, v6, v7 = featureWPT(
        imageData[i])
    featureStack = np.hstack((rms, mav, wl, zc, ssc, var, wa, me, mcv, mfp, ar0, ar1, ar2, ar3, ar4, ar5, ar6, sm2, mf,
                              e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5,
                              v6, v7))
    featureData.append(featureStack)
    featureLabel.append(imageLabel[i])
featureData = np.array(featureData)

# print(featureData.shape)
# print(len(featureLabel))


# import time
# start_time = time.time()
train_x, test_x, train_y, test_y = train_test_split(featureData, featureLabel, test_size=0.2)
"""
n_estimators : (default=10) The number of trees in the forest.
max_features: 寻找最佳分割时要考虑的特征数量
bootstrap: 默认True，构建树时是否使用bootstrap样本。

"""
RF = RandomForestClassifier(n_estimators=180, criterion='gini', max_depth=None, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0,
                            warm_start=False, class_weight=None)

RF.fit(train_x, train_y)
score = RF.score(train_x, train_y)
predict = RF.predict(test_x)
accuracy = metrics.accuracy_score(test_y, predict)

print("RF train accuracy: %.2f%%" % (100 * score))
print('RF test  accuracy: %.2f%%' % (100 * accuracy))
# print ('training took %fs!' % (time.time() - start_time))


# ser = serial.Serial('com6', 115200)
while (True):  # 30可以根据需要设置，while(True)：代表一直读下去
    data_recongition = []
    featureData2 = []
    count = 0
    while count != 200:  # 30可以根据需要设置，while(True)：代表一直读下去
        data = ser.readline()  # 按行读取串口数据进来
        try:
            # reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER
            data = data.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
            data = data[:-2]
            data = data.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
            data = list(map(float, data))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
            # print(data)
            if np.any(pd.DataFrame(data).isnull()):
                pass
            else:
                data_recongition.append(data)  # 添加到列表里
                # print(count)
                count += 1
        except:
            pass
    data_recongition = pd.DataFrame(data_recongition)
    data_recongition = np.array(data_recongition)

    timeWindow = 200
    strideWindow = 200
    # for i in tqdm(range(2)):
    rms = featureRMS(data_recongition)
    mav = featureMAV(data_recongition)
    wl = featureWL(data_recongition)
    zc = featureZC(data_recongition)
    ssc = featureSSC(data_recongition)
    var = featureVAR(data_recongition)
    wa = featureWA(data_recongition)
    me = featureME(data_recongition)
    mcv = featureMCV(data_recongition)
    mfp = featureMPF(data_recongition)
    ar0, ar1, ar2, ar3, ar4, ar5, ar6 = featureAR(data_recongition)
    sm2 = featureSM2(data_recongition)
    mf = featureMF(data_recongition)
    e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5, v6, v7 = featureWPT(
        data_recongition)
    featureStack = np.hstack((rms, mav, wl, zc, ssc, var, wa, me, mcv, mfp, ar0, ar1, ar2, ar3, ar4, ar5, ar6, sm2, mf,
                              e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5,
                              v6, v7))
    featureData2.append(featureStack)
    # featureLabel.append(imageLabel[i])
    featureData2 = np.array(featureData2)
    predict = RF.predict(featureData2)
    print(predict)