""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from scipy.io import wavfile
from sklearn import metrics
import matplotlib.pyplot as plt
import math 
from scipy.fftpack import fft,ifft
import soundfile
import winsound
import sprt as sprt
from detecta import detect_cusum
import pandas as pd
import seaborn as sns
from scipy.stats import *
from sklearn.neighbors.kde import KernelDensity
from statsmodels.stats.diagnostic import normal_ad
import scipy.stats as stats
from time import time
""" ======================  Function definitions ========================== """
def smooth(x, win):
    step = len(x)//win
    remain = len(x)%win
    y = np.zeros(len(x))
    for i in range(step):
        mean = x[i*win:(i + 1)*win].mean()
        y[i*win:(i + 1)*win] = mean
    y[len(x) - remain::] = x[len(x) - remain::].mean()
    return y

def esmooth(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp

def LMS(x, d, ss, order):
    step = len(x)
    order = int(order)
    w = np.zeros((order,step))
    X = np.zeros((order,step))
    e = np.zeros(step)
    y = np.zeros(step)
    allmse = np.zeros(step)
    tempx = x
    for i in range(order):
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:step]
        X[i,:] = tempx

    for i in range(3, step - 1):
        y[i] = np.dot(w[:,i],X[:,i])
        e[i] = d[i] - y[i]
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * ss/(0.1 + l**2) * e[i] * X[:,i]
        allmse[i + 1] = np.sum(e**2)/(i + 1)
    mse = np.mean(np.power(e,2))
    return w[:,step - 1].tolist()

def LMStrain(x, d, ss, order):
    step = len(x)
    w = np.zeros((order,step))
    X = np.zeros((order,step))
    e = np.zeros(step)
    y = np.zeros(step)
    allmse = np.zeros(step)
    tempx = x
    for i in range(order):
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:step]
        X[i,:] = tempx

    for i in range(step - 1):
        y[i] = np.dot(w[:,i],X[:,i])
        e[i] = d[i] - y[i]
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * ss/(0.1 + l**2) * e[i] * X[:,i]
        #allmse[i + 1] = np.sum(e**2)/(i + 1)
    mse = np.mean(np.power(e,2))
    return mse

def LMStest(t, d, w, order, win):
    step = len(t)
    order = int(order)
    X = np.zeros((order,step))
    e = np.zeros(step)
    y = np.zeros(step)
    allmse = np.zeros(step)
    tempx = t
    for i in range(order):
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:step]
        X[i,:] = tempx

    for i in range(3, step - 1):
        y[i] = np.dot(w,X[:,i])
        e[i] = d[i] - y[i]
        #allmse[i + 1] = np.sum(e**2)/(i + 1)
    mse = np.mean(np.power(e,2))
    e = smooth(abs(e), win)
    return e

def findss(x, d, order):
    mse1 = np.zeros(10)
    mse2 = np.zeros(20)
    mse3 = np.zeros(20)
    for i in range(10):
        mse1[i] = LMStrain(x, d, 0.1*i, order)
    idx1 = np.argmin(mse1)
    for i in range(20):
        mse2[i] = LMStrain(x, d, (idx1/10 - 0.1) + 0.01*i, order)
    idx2 =  np.argmin(mse2)
    for i in range(20):
        mse3[i] = LMStrain(x, d, (idx1/10 - 0.1) + (idx2/100 - 0.01) + 0.001*i, order)
    idx3 =  np.argmin(mse3)
    bestss = (idx1/10 - 0.1) + (idx2/100 - 0.01) + idx3/1000
    return bestss

def findod(x, d, ss):
    mse = np.zeros(20)
    for i in range(20):
        mse[i] = LMStrain(x, d, ss, i + 1)
    index = np.argmin(mse)
    bestod = index + 1
    return bestod

def plotswave(window):
    y = np.zeros(len(data_test))
    E = np.zeros((11,len(data_test)))
    yout_signal = []
    yout_noise = []

    for i in range(10):
        E[i,:] = LMStest(data_test, data_test, w_train[i], od[i], window)
    E[10,:] = LMStest(data_test, data_test, w_noise, od[10], window)

    for i in range(len(data_test)):
        if np.argmin(abs(E[:,i])) == 10:
            y[i] = 0
            
        else:
            y[i] = 1
    return y

def em_single(priors,observations):

    """
    EM算法的单次迭代
    Arguments
    ------------
    priors:[theta_A,theta_B]
    observation:[m X n matrix]

    Returns
    ---------------
    new_priors:[new_theta_A,new_theta_B]
    :param priors:
    :param observations:
    :return:
    """
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]
    #E step
    for observation in observations:
        len_observation = len(observation.tolist())
        num_heads = observation.sum()
        num_tails = len_observation-num_heads
        #二项分布求解公式
        contribution_A = scipy.stats.binom.pmf(num_heads,len_observation,theta_A)
        contribution_B = scipy.stats.binom.pmf(num_heads,len_observation,theta_B)

        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        #更新在当前参数下A，B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    # M step
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A,new_theta_B]

def em(observations,prior,tol = 1e-6,iterations=10000):
    """
    EM算法
    ：param observations :观测数据
    ：param prior：模型初值
    ：param tol：迭代结束阈值
    ：param iterations：最大迭代次数
    ：return：局部最优的模型参数
    """
    iteration = 0;
    while iteration < iterations:
        new_prior = em_single(prior,observations)
        delta_change = numpy.abs(prior[0]-new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration +=1
    return [new_prior,iteration]

def autocorr(a, order, window):
    r = np.zeros(order)
    x = np.zeros((order,window))
    tempa = a[0:window]
    for i in range(order):
        x[i,:] = tempa
        tempa = np.insert(tempa,0,0)
        tempa = tempa[0:window]
    R = np.dot(x,x.T)
    return R
""" ====================== Variable Declaration ========================== """

""" ========================= Input Generation =========================== """
rate_train, data_train = wavfile.read('train_signal.wav')
rate_noise, data_noise = wavfile.read('noise_signal1.wav')
rate_test, data_test = wavfile.read('test_signal1.wav')
data_train = data_train[0:1203440]
data_train10 = data_train.reshape(10, 120344)
data_train10 = data_train10.tolist()
data_train1 = []

for i in range(10):
    temp = []
    for j in range(120344):
        if data_train10[i][j] != 0 and data_train10[i][j] != 1:
            temp.append(data_train10[i][j])
    data_train1.append(temp)

'''f1, t1, z1 = signal.stft(data_train, fs = 48000)
plt.pcolormesh(t1, f1, abs(z1), vmax = 100)
plt.xlabel('Manatee Calls')
plt.ylabel('Frequency(Hz)')
plt.colorbar()
plt.show()
'''


y = np.zeros(1440000)
y[57000] = 1
y[117000] = 1
y[170000] = 1
y[239000] = 1
y[279000] = 1
y[382000] = 1
y[414000] = 1
y[470000] = 1
y[563000] = 1
y[728000] = 1
y[751000] = 1
y[890000] = 1
y[947000] = 1
y[993000] = 1
y[1209000] = 1
y[1239000] = 1

plt.plot(y)
plt.xlabel('Samples')
plt.show()'''

'''x = [0, 0, 1, 1, 2, 3, 9]
y = [0, 13, 13, 14, 14, 15, 15]
plt.xlim((0,9))
plt.ylim((0,16))
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.plot(x, y)
plt.show()
""" ============================ A ===================================== """
window = 1000

w_train = []
y_train = np.zeros((10,len(data_test)))
ss = np.zeros(11)
od = np.zeros(11)
for i in range(10):
    ss[i] = findss(np.array(data_train1[i]), np.array(data_train1[i]), 3)
ss[10] = findss(data_noise, data_noise, 3)

for i in range(10):
    od[i] = findod(np.array(data_train1[i]), np.array(data_train1[i]), ss[i])
od[10] = findod(data_noise, data_noise, ss[10])

start1 = time()
for i in range(10):
    w_train.append(LMS(np.array(data_train1[i]), np.array(data_train1[i]), ss[i], od[i]))

w_noise = LMS(data_noise, data_noise, ss[10], od[10])

y = np.zeros(len(data_test))
E = np.zeros((11,len(data_test)))
yout_signal = []
yout_noise = []

for i in range(10):
    E[i,:] = LMStest(data_test, data_test, w_train[i], od[i], window)
E[10,:] = LMStest(data_test, data_test, w_noise, od[10], window)

for i in range(len(data_test)):
    if np.argmin(abs(E[:,i])) == 10:
        y[i] = 0
        yout_noise.append(data_test[i])
    else:
        y[i] = 1
        yout_signal.append(data_test[i])
end1 = time()
soundfile.write('output_signal.wav', yout_signal, 48000)
soundfile.write('output_noise.wav', yout_noise, 48000)

y1 = plotswave(100)
y2 = plotswave(500)
y3 = plotswave(600)
y4 = plotswave(800)

plt.figure(1)
plt.plot(y1)
plt.xlabel('Samples')

plt.figure(2)
plt.plot(y2)
plt.xlabel('Samples')

plt.figure(3)
plt.plot(y3)
plt.xlabel('Samples')

plt.figure(4)
plt.plot(y4)
plt.xlabel('Samples')

plt.show()

""" ============================ B ===================================== """
test = sprt.SPRTBinomial(h0 = 0.5, h1 = 0.5, alpha = 0.05, beta = 0.2, values = data_noise)
test.plot()'''

'''sns.kdeplot(data_noise)
plt.show()'''
'''a = autocorr(data_test, 1, len(data_test))
plt.plot(a)
plt.show()
a = em(data_noise, [0.6, 0.5])
a = stats.ks_2samp(data_test[0:100], data_noise)

auto1 = np.correlate([1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1,1], mode = 'full')

plt.plot(auto1)

#plt.plot(data_train1[0])
plt.show()


plt.plot(y)
plt.xlabel('Samples')
plt.show()



x1 = [0, 0, 3, 7, 9]
y1 = [0, 8, 12, 13, 14]
x2 = [0, 0, 1, 1, 2, 3, 9]
y2 = [0, 13, 13, 14, 14, 15, 15]
plt.xlim((0,9))
plt.ylim((0,16))
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.plot(x1, y1, label = 'CUSUM')
plt.plot(x2, y2, label = 'NLMS')
plt.legend()
plt.show()

start2 = time()
ta, tai, taf, amp = detect_cusum(data_test, 2000, 10000, True, True)
end2 = time()

time1 = end1 - start1
time2 = end2 - start2

x1 = ['NLMS', 'CUSUM']
y1 = [time1, time2]
plt.bar(x1,y1)
plt.ylabel("Time(sec)")
plt.show()
