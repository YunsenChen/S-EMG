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
from datetime import datetime
import math
from feature_utils import *
from tqdm import tqdm
import os
from joblib import dump, load

# sets up serial connection (make sure baud rate is correct - matches Arduino)
# 设置串口号和波特率和Arduino匹配
ser = serial.Serial('com3', 115200)

def readData(count,threshold=1000):
    gestureData=[]
    while count>0:
        line = ser.readline()  # 按行读取串口数据进来
        #print("读取进来:",line)
        # reads until it gets a carriage return. MAKE SURE THERE IS A CARRIAGE RETURN OR IT READS FOREVER
        try:
            line = line.decode()  # 读进来的数据是bytes形式，需要转化为字符串格式
            #print("decode后:",line)
            line = line[:-2]
            #print("截断后:",line)
            line = line.split(",")  # 以空格为分隔符分隔字符串-->['180.87', '2.16', '-3.86']
            #print("split后:",line)
            line = list(map(float, line))  # 把字符串转化为数字-->[180.87, 2.16, -3.86]
            #print("转为数字后:",line)
            # print(data)
            if None in line:
                print("front is null")
            else:
                for i in line:
                    if abs(i) > threshold:
                        print("肌电信号大于设定阈值:",threshold)
                    else:
                        gestureData.append(line)  # 添加到列表里
                        #print(count)
                        count -= 1
        except:
            #print("except被执行")
            pass
    return gestureData




def collectData(gestureGroup=['front','back','left','right'],samples_per_gesture=2000,gestures_repeat=4):
    data={}
    for gesture in gestureGroup:
        data[gesture] = []  # 为每个手势名称创建一个空数组
    for _ in range(gestures_repeat):
        for gesture in gestureGroup:
            print(gesture)
            time.sleep(1)
            gestureData=readData(count=samples_per_gesture)
            data[gesture]=data[gesture]+gestureData
    return data



def saveDataToExcel(data):
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 创建一个 ExcelWriter 对象，指定文件名
    file_name = f"data_{current_time}.xlsx"
    # 检查是否存在 data 目录，如果不存在则创建
    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = os.path.join("data", file_name)
    with pd.ExcelWriter(file_path) as writer:
        for key, value in data.items(): 
            df_emg = pd.DataFrame(value)
            df_emg.to_excel(writer, sheet_name=key, index=False)


def normalizeData(data):
    emg = []
    label = []
    labelCount=0 #逐步加1
    #print(len(data['front']))
    for key, value in data.items():
        #print("key:",key)
        #print("value:",len(value))
        label_value=np.full(len(value), labelCount)
        emg.extend(value)
        label.extend(label_value)
        labelCount+=1
    emg=np.array(emg)
    return emg,label


def convertDataToImageData(emg,label,timeWindow=200,strideWindow=200):
    #print("emg:",emg)
    #print("emg.shape:",emg.shape)
    #print("len(emg):",len(emg))
    # print("len(label):",len(label))
    #print("label:",label)
    classes=max(label)+1
    timeWindow = 200
    strideWindow = 200
    imageLabel=[]
    imageData=[]
    for i in range(classes):
        index=[]
        for j in range(len(label)):
            if(label[j]==i):
                index.append(j)

        iemg=emg[index,:]
        length=math.floor((iemg.shape[0]-(strideWindow-timeWindow))/timeWindow)
        print("class ", i, " number of sample: ", iemg.shape[0], length)

        for j in range(length):
            subImage = iemg[timeWindow * j:strideWindow * (j + 1), :]
            imageData.append(subImage)
            imageLabel.append(i)
    imageData = np.array(imageData)
    return imageData,imageLabel

def featureStackForImageData(imageData):
    rms = featureRMS(imageData)
    mav = featureMAV(imageData)
    wl = featureWL(imageData)
    zc = featureZC(imageData)
    ssc = featureSSC(imageData)
    var = featureVAR(imageData)
    wa = featureWA(imageData)
    me = featureME(imageData)
    mcv = featureMCV(imageData)
    mfp = featureMPF(imageData)
    ar0, ar1, ar2, ar3, ar4, ar5, ar6 = featureAR(imageData)
    sm2 = featureSM2(imageData)
    mf = featureMF(imageData)
    e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5, v6, v7 = featureWPT(
        imageData)
    featureStack = np.hstack(
        (rms, mav, wl, zc, ssc, var, wa, me, mcv, mfp, ar0, ar1, ar2, ar3, ar4, ar5, ar6, sm2, mf,
            e0, e1, e2, e3, e4, e5, e6, e7, a0, a1, a2, a3, a4, a5, a6, a7, v0, v1, v2, v3, v4, v5,
            v6, v7))
    
    return featureStack
    

def featureExtraction(imageData,imageLabel):
    featureData = []
    featureLabel = []
    for i in tqdm(range(imageData.shape[0])):
        try:
            featureStack=featureStackForImageData(imageData[i])
            featureData.append(featureStack)
            featureLabel.append(imageLabel[i])
        except:
            pass 

    featureData = np.array(featureData)
    return featureData,featureLabel

def saveModel(rf_model,accuracy):
    save_dir = "model"
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 创建一个 ExcelWriter 对象，指定文件名
    file_name = f"rf_acc_{accuracy}_{current_time}.joblib"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存模型
    model_path = os.path.join(save_dir, file_name)
    dump(rf_model, model_path)

    print(f"Model saved successfully at {model_path}")


data=collectData()
saveDataToExcel(data)
emg,label=normalizeData(data)
imageData,imageLabel=convertDataToImageData(emg,label)
featureData,featureLabel=featureExtraction(imageData,imageLabel)
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
saveModel(RF,accuracy)
print("RF train accuracy: %.2f%%" % (100 * score))
print('RF test  accuracy: %.2f%%' % (100 * accuracy))
# print ('training took %fs!' % (time.time() - start_time))

while (True):  # 30可以根据需要设置，while(True)：代表一直读下去
    featureData2 = []
    count = 0
    dataTorecongition=readData(count=200)
    dataTorecongition = pd.DataFrame(dataTorecongition)
    dataTorecongition = np.array(dataTorecongition)
    featureStack=featureStackForImageData(dataTorecongition)
    
    # featureLabel.append(imageLabel[i])
    if len(featureStack)!=0:
        featureData2.append(featureStack)
        featureData2 = np.array(featureData2)
        predict = RF.predict(featureData2)
        print(predict)