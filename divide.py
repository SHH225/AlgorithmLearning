import cv2
import random
from matplotlib import pyplot as plt
#将二进制数扩展成8位（不足八位的数在前面补零）
def expand(seed):
    for i in range(len(seed)):
        seed[i] = seed[i][:2] + '0' * (10-len(seed[i])) + seed[i][2:len(seed[i])]
    return seed
#获取每个像素值的概率和均值
def Hist(image):
    a = [0] * 256
    h = image.shape[0]
    w = image.shape[1]
    n = h * w
    average = 0.0
    for i in range(h):
        for j in range(w):
            pixel = int(image[i][j])
            a[pixel] = a[pixel] + 1
    for i in range(256):
        a[i] = a[i] / float(n)
        average = average + i*a[i]
    return a, average
#获取阈值
def getTh(seed):
    th = [0,256]
    seedInt = int(seed,2)
    tmp = seedInt & 255
    th.append(tmp)
    th.sort()
    return th
#得到适应度值 越大越好
def fitness(seed, p ,average):
    Var = [0.0] * len(seed)
    for i in range(len(seed)):
        th = getTh(seed[i])
        w = [0.0] * 2
        muT = [0.0] * 2
        mu = [0.0] * 2
        for j in range(2):
            for k in range(th[j],th[j+1]):
                w[j] = w[j] + p[k]
                muT[j] = muT[j] + p[k] * k
            if w[j] > 0:
                mu[j] = muT[j] / w[j]
                Var[i] = Var[i] + w[j] * pow(mu[j] - average,2)
    return Var
#自然选择（轮盘赌算法）                  
def wheel_selection(seed,Var):    
    var = [0.0] * len(Var)
    n = [''] * len(seed)
    sumV = sum(Var)
    s = []
    for i in range(len(Var)):
        var[i] = Var[i] / sumV
    for i in range(1,len(Var)):
        var[i] = var[i] + var[i-1]
    for i in range(len(seed)):
        s.append(random.random()) 
    s.sort()
    fitin = 0
    newin = 0
    while newin < len(seed):
        if s[newin] < var[fitin]:
            n[newin] = seed[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1            
    return n
 #个体间交叉，实现基因交换 
def Cross(Next,crossP):
    for i in range(len(Next)-1):
        if random.random() < crossP:
            n = random.randint(2,9)
            tmp1 = Next[i]
            tmp2 = Next[i+1]
            Next[i] = tmp1[:n] + tmp2[n:]
            Next[i+1] = tmp2[:n] + tmp1[n:]
    return Next
 #基因突变  
def Variation(Next,vp):
    for i in range(len(Next)):
        if random.random() < vp:
            n = random.randint(2,9)
            tmp = Next[i]
            if tmp[n] == '0':
                Next[i] = Next[i][:n] + '1' + Next[i][n+1:]
            if tmp[n] == '1':
                Next[i] = Next[i][:n] + '0' + Next[i][n+1:]
    return Next
#用openCV库读取图片的灰度值
image = cv2.imread('D:\\boy.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#随机生成初代种群
items_x = list(range(0,20))
items_y = list(range(0,20))
random.shuffle(items_x)
random.shuffle(items_y)
x = items_x[0:20]
y = items_y[0:20]
seed = []
times = 0
#交叉率
cp = 0.7
#变异率
vp = 0.06
for i in range(0,20):
    code = gray[x[i]][y[i]]
    seed.append(bin(code))
#获取每个像素值的概率和均值
p, average = Hist(gray)
Varpre = fitness(seed, p, average)
 
while times < 2000:
    Var = fitness(seed, p, average)
    Next = wheel_selection(seed,Var)
    Next = expand(Next)
    Next = Cross(Next,cp)
    Next = Variation(Next,vp)
    seed = Next
    times = times +1
    
for j in range(len(Var)):
    if Var[j] == max(Var):
        k = getTh(Next[j])
        
print(k)
#阈值分割后每个像素点的值
def genetic_thres(image, k):
    th = image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > k[0] and image[i][j] < k[1]:
                th[i][j] = int(k[0])
    return th

plt.subplot(221), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(image.ravel(), 256,[0,256])
plt.title("Histogram")
th1 = genetic_thres(gray, k)
plt.subplot(222), plt.imshow(th1, "gray")
plt.title("threshold is " + str(k[1])), plt.xticks([]), plt.yticks([])
plt.show()