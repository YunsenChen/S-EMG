import numpy as np
import numpy.fft as fft
from scipy import signal
import pywt
from statsmodels.tsa.ar_model import AutoReg

#均方根
def featureRMS(data):
    return np.sqrt(np.mean(data**2, axis=0))

#平均绝对值
def featureMAV(data):
    return np.mean(np.abs(data), axis=0)

#波形长
def featureWL(data):
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

#零穿越次数
def featureZC(data, threshold=10e-7):
    numOfZC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(1,length):
            diff = data[j,i] - data[j-1,i]
            mult = data[j,i] * data[j-1,i]
            
            if np.abs(diff)>threshold and mult<0:
                count=count+1
        numOfZC.append(count/length)
    return np.array(numOfZC)

#斜率符号变化次数
def featureSSC(data,threshold=10e-7):
    numOfSSC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(2,length):
            diff1 = data[j,i]-data[j-1,i]
            diff2 = data[j-1,i]-data[j-2,i]
            sign  = diff1 * diff2
            
            if sign>0:
                if(np.abs(diff1)>threshold or np.abs(diff2)>threshold):
                    count=count+1
        numOfSSC.append(count/length)
    return np.array(numOfSSC)

#方差
def featureVAR(data):
    return np.var(data,axis=0)

#Willision幅值WA
def featureWA(data,threshold=10e-7):
    return sum(np.where(np.diff(data, axis=0) > threshold, 1, 0))/data.shape[0]


#幅值立方均值
def featureMCV(data):
    return sum(data**3)/data.shape[0]

#平均功率频率
def featureMPF(data):
    fs=2000
    Pxx_all = []
    for i in range(data.shape[1]):
        f, Pxx = signal.welch(data[:, i], fs, nperseg=200)
        mean_power = np.trapz(Pxx, f)
        mean_frequency = np.trapz(f * Pxx, f) / mean_power
        Pxx_all.append(mean_frequency)
    Pxx_all = np.array(Pxx_all)
    return Pxx_all

#对数检测值logD


#自回归模型系数ARC
def featureAR(data):
    AR_num = 7
    row = []
    for j in range(data.shape[1]):
        x = data[:, j]
        AR7_model = AutoReg(x, AR_num).fit()
        AR_params = AR7_model.params[1:AR_num + 1]
        row.append(AR_params)
    row=np.array(row)
    return row[:,0],row[:,1],row[:,2],row[:,3],row[:,4],row[:,5],row[:,6]


#二阶谱矩
def featureSM2(data):
    fs=2000
    Pxx_all = []
    for i in range(data.shape[1]):
        f, Pxx = signal.welch(data[:, i], fs, nperseg=200)
        #mean_power = np.trapz(Pxx, f)
        mean_frequency = np.trapz(f*f * Pxx, f)
        Pxx_all.append(mean_frequency)
    Pxx_all = np.array(Pxx_all)
    return Pxx_all


#中值频率f_md
def featureMF(data):
    fs=2000
    Pxx_all = []
    for i in range(data.shape[1]):
        f, Pxx = signal.welch(data[:, i], fs, nperseg=200)
        mean_power = np.trapz(Pxx, f)
        #mean_frequency = np.trapz(f * Pxx, f) / mean_power
        Pxx_all.append(mean_power/2)
    Pxx_all = np.array(Pxx_all)
    return Pxx_all



#均值频率f_me
def featureME(data):
    return np.mean(abs(fft.fft(data)), axis=0)


#短时傅里叶变换SFT


#小波变换WT



def featureWPT(data):
    e=[]
    a=[]
    v=[]
    for i in range(data.shape[1]):
        wp = pywt.WaveletPacket(data=data[:,i], wavelet='db3',mode='symmetric',maxlevel=3)
        n = 3
        re = []  #第n层所有节点的分解系数
        for i in [node.path for node in wp.get_level(n, 'freq')]:
            re.append(wp[i].data)
        #第n层能量特征
        energy = []
        for i in re:
            energy.append(pow(np.linalg.norm(i,ord=None),2))
        average=[]
        #第n层均值
        for i in re:
            average.append(np.average(i))
        var=[]
        for i in re:
            var.append(np.var(i))
        a.append(average)
        e.append(energy)
        v.append(var)
    a=np.array(a)
    e=np.array(e)
    v=np.array(v)
    return e[:,0],e[:,1],e[:,2],e[:,3],e[:,4],e[:,5],e[:,6],e[:,7],a[:,0],a[:,1],a[:,2],a[:,3],a[:,4],a[:,5],a[:,6],a[:,7],v[:,0],v[:,1],v[:,2],v[:,3],v[:,4],v[:,5],v[:,6],v[:,7]





