import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import scipy.ndimage as ndimage
from scipy import signal


def plotSignalAndFt(t, signal, fs):
    ''' Plots the input signal and the fourier transformed signal with respect to the time.
          t: time
          signal: input signal
          fs: fourier transformed signal'''
    plt.figure(figsize=(16, 4))
    sp1 = plt.subplot(121)
    sp1.set_xlim(0, 1)
    sp1.grid(True)
    sp1.set_xlabel('x')
    sp1.set_ylabel('Amplitude')
    sp1.set_title('Spacial Domain')
    sp1.plot(t, signal, 'bx--', markersize=6)

    sp2 = plt.subplot(122)
    sp2.set_xlabel('Fourier Transform')
    sp2.set_ylabel('Amplitude')
    sp2.set_title('Frequency Domain')
    sp2.grid(True)
    fftLength = np.array(fs.shape)
    sp2.set_xlim(0, fftLength - 1)
    # 注意使用绝对值，否则会只是用实部画图
    sp2.plot(np.arange(fftLength), abs(fs), 'bx', markersize=14)
    plt.show()

n = 100
k = 2
eingabe = np.cos(k* np.linspace(0,2*np.pi, n,endpoint=False))
y = np.fft.fft(eingabe)
plotSignalAndFt(n, 2,y)
print(np.where(np. abs(y)>0.1))



'''
def fill_box(f):
    Kernel_ver = np.zeros(shape=[3, 3])
    Kernel_ver[:, 1] = 0
    Kernel_quad = np.ones((3, 3))

    result_closing = closing(f, Kernel_ver)

    result_filled = fill_holes(result_closing)

    result_opening = opening(result_filled, Kernel_quad)
    return
'''
'''
# code buffer 1
def c1assifyF1agImage(img, f1agCo1ors):
ny, nx, nc = img.shape # nc = 3 is the number of co1ors
number0OfFlags, number0OfStripes, nc = flagColors.shape# we just assume the input dimension is correct - no need to check
#Continue here:
'''

'''
# code buffer 2
A = np.array([[9, 7, 1, 4, 1],
              [2, 7, 6, 1, 3],
              [7, 0, 10, 9, 8],
              [1, 4, 1, 1, 5],
              [6, 9, 8, 6, 9]])

def medianFilter(img):
    ny, nx = img.shape
    #Der als Antwort gegebene Code kommt hier hin
    filteredImg = np.zeros((nx-2, ny-2))
    for i in range(0, nx-2):
        for j in range(0, ny-2):
            filteredImg[i, j] = np.floor(np.median(img[i:i + 3, j:j + 3]))
    return filteredImg

print(medianFilter(A))

'''


'''
# K5. Fourier Transformation 傅里叶变换
# 使用fft.fft2等函数进行变换
img_dir = "test.jpg"
img_pil = Image.open(img_dir)
img = np.array(img_pil.resize([256, 256]))/255
img = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3

# Plot
plt.figure(figsize=(5,5))
plt.imshow(img, 'gray')
plt.show()

# FFT
img_fft = np.fft.fftshift(np.fft.fft2(img))
plt.figure(figsize=(5, 5))
# 记得np.log，拓展坐标
plt.imshow(np.log(np.absolute(img_fft)))
plt.show()

ny, nx = img.shape
x = np.linspace(-1.5, 1.5, nx)
y = np.linspace(-1.5, 1.5, ny)
xv, yv = np.meshgrid(x, y)
mask = 1 + np.sqrt(xv**2 + yv**2)
plt.figure(figsize=(5, 5))
plt.imshow(mask, vmin=0, vmax=np.amax(mask))
plt.colorbar()
plt.show()

# 扩展高频，降低低频
img_fft_manipulated = mask*img_fft
# 逆傅里叶变换， 最后使用real部， np.real
img_new = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_manipulated)))
plt.figure(figsize=(15,10))
sp1 = plt.subplot(121)
sp1.imshow(img, 'gray', vmin=0, vmax=1)
sp2 = plt.subplot(122)
sp2.imshow(img_new, 'gray', vmin=0, vmax=1)
plt.show()

### 卷积部分
# 时域卷积等于频域相乘
def convolution_via_fft(x,y):
    fft_x = np.fft.fft2(x)
    fft_y = np.fft.fft2(y)
    return np.fft.ifft2(fft_x*fft_y)

kernel = np.array([[0, -1.0, 0], [-1.0, 8.0, -1.0], [0, -1.0, 0]])/4
kernelBig = np.zeros([ny,nx])

# 获得中心点坐标，和经过逆shift运算后的核
mid_y = np.int(np.floor(ny/2.0))
mid_x = np.int(np.floor(nx/2.0))
kernelBig[mid_y-1:mid_y+2, mid_x-1:mid_x+2] = kernel
kernelBig = np.fft.ifftshift(kernelBig)

# F(sharpened1) = F(img) * F(kernel)
sharpened1 = signal.convolve2d(img, kernel, boundary='wrap', mode='same')  # 卷积两个二维数组
sharpened2 = np.real(convolution_via_fft(img, kernelBig))  # 注意使用kernelBig，因为涉及到了函数运算，方便计算

# Plot
plt.figure(figsize=(5,5))
plt.imshow(sharpened2-sharpened1)
plt.colorbar()
plt.show()

# 或者反过来，我们将如何解决以下问题：
# 给定“ sharpended1”和“ kernel”，以及通过将图像“ img”与“ kernel”折叠来创建图像“ sharpened”的信息，请确定如何我们“ img”？
ny, nx = sharpened1.shape
kernelBig = np.zeros([ny,nx])

# 获得中心点坐标，和经过逆shift运算后的核
mid_y = np.int(np.floor(ny/2.0))
mid_x = np.int(np.floor(nx/2.0))
kernelBig[mid_y-1:mid_y+2, mid_x-1:mid_x+2] = kernel  # 注意使用kernelBig
kernelBig = np.fft.ifftshift(kernelBig)

# F(img) = F(sharpened1) / F(kernel)
fft_sharpened1 = np.fft.fft2(sharpened1)
fft_kernelBig = np.fft.fft2(kernelBig)
recoveredImg = np.real(np.fft.ifft2(fft_sharpened1/fft_kernelBig))

plt.figure(figsize=(5,5))
plt.imshow(img-recoveredImg)
plt.colorbar()
plt.show()
'''

'''
# K7. 非线性滤波 - 线性滤波器的缺点:图像结构缺失
n = 100
x = np.arange(n)
f = np.array(x<n/2.0, dtype='float')
f_noisy = f + np.random.normal(0, .1, f.shape)

plt.figure(figsize=(16, 4))
plt.plot(x,f,'--')
plt.xlim((0, n-1))
plt.plot(x,f_noisy,'-')
plt.ylim((-1.2, 1.2))
plt.show()

bet = .1
w = 4
smoothing_kernel = np.exp(-bet*np.arange(-w, w+1)**2)
# 将滤波器的核归一化
smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)
print(smoothing_kernel)
f_noisy_extended = np.pad(f_noisy, w, mode='edge')

plt.figure(figsize=(16, 4))
plt.plot(x,f,'--', )
# 使用1d相关滤波器
plt.plot(x,np.correlate(f_noisy_extended, smoothing_kernel, mode='valid'),'-')
plt.xlim((0, n-1))
plt.ylim((-1.2, 1.2))
plt.show()
'''


'''
# K7. 非线性滤波 Noise an Histograms
def gaussian(mu, sigma, x):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

mu, sigma = 0, 1  # 平均值， 标准差
noise = mu + sigma * np.random.randn(10000)

# plot the histogram of the noise 直方图
n, bins, patches = plt.hist(noise, 50, density=True, facecolor='g', alpha=0.75)

x = np.linspace(mu-4*sigma,mu+4*sigma, 1000)

# plot Gaussian normal curve with corresponding mean and std. 高斯钟形曲线
plt.plot(x, gaussian(mu,sigma,x), '--', linewidth=4)

plt.xlabel('Numbers')
plt.ylabel('Probability')
plt.title('Histogram')
plt.grid(True)
plt.show()
'''

'''
# K7 中值滤波器
A = np.array([[9, 7, 1, 4, 1],
              [2, 7, 6, 1, 3],
              [7, 0, 10, 9, 8],
              [1, 4, 1, 1, 5],
              [6, 9, 8, 6, 9]])
B = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        B[i, j] = np.floor(np.median(A[i:i+3, j:j+3]))
print(B)
'''

'''
# K8. 测试高斯核
A = np.arange(9).reshape((3, 3))
A= A / np.max(A)
B = A**2
test = np.zeros_like(A)
sigma_int = 10
w = 2
test = np.exp(-sigma_int * (B - A)**2)
print(test)
'''


'''
# K9. 图像分割，用于获取图像四边形的坐标点。
img = np.arange(2500).reshape([50, 50])
img = img/np.max(img)
# 制作一个fake彩图
img = np.stack((img, np.rot90(img, 1), np.rot90(img, 2)), axis=-1)
A = np.zeros((50, 50), dtype='int')
A[5:30, 20:50] = 1

ny, nx = A.shape
xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))

plt.imshow(A)
plt.show()

# 需要置于远处，在后面需要计算
xv[A==0] = -nx
yv[A==0] = -ny


# 图像的四个坐标点
corners = np.array([[0, 0], [0, nx-1], [ny-1, 0], [ny-1, nx-1]])
# 四边形角的索引
ind = -np.ones([4, 2])

# 计算哪个点距离四个边角最小，使用曼哈顿距离
for i in range(0, 4):
    temp = np.abs(np.array(xv) - corners[i, 1]) + np.abs(np.array(yv) - corners[i, 0])
    # 获得最小点的平面1d索引
    test = np.argmin(temp, axis=None)
    # 获得最小点的索引并写入变量中。np.unravel_index将平面索引或平面索引数组转换为坐标数组的元组。
    ind[i, :] = np.unravel_index(np.argmin(temp, axis=None), temp.shape)

# 在每个图层上应用矩形
plt.imshow(np.stack((A,)*3, axis=-1)*img)
for i in range(0, 4):
    # 在四个点上画×
    plt.plot(ind[i, 1], ind[i, 0], 'gx', markersize=22, markeredgewidth=4, color='red')
plt.show()
'''

'''
# K9. binary_fill_holes 填补孔洞
def simpleFillHoles(binaryImg):
    # 建立一个和图像相同的全零矩阵
    temp = np.zeros(binaryImg.shape, bool)

    # 建立结构元素
    structureElement = np.ones((3, 3), bool)
    structureElement[0, 0] = False
    structureElement[0, 2] = False
    structureElement[2, 0] = False
    structureElement[2, 2] = False

    # 主循环
    while True:
        tempold = temp.copy()
        # border_value=1 输出数组中边框处的值
        temp = ndimage.morphology.binary_dilation(temp, structureElement, border_value=1, origin=0)

        # 将原图像置0
        temp[binaryImg] = False

        # 检查图像
        if (temp == tempold).all():  # 如果图像不发生变化
            break

        # Plot
        plt.imshow(temp, 'gray')
        plt.show()
        plt.pause(0.3)
    return ~temp


# 定义初始图像
binaryImg = np.zeros((15, 15), dtype=bool)
binaryImg[1:5, 3:7] = 1
binaryImg[2:3, 4:5] = 0
binaryImg[7:13, 2:13] = 0
binaryImg[9:12, 4:11] = 1
# print(binaryImg.astype(int))

plt.imshow(binaryImg)
plt.show()

filled = simpleFillHoles(binaryImg)
'''

'''
# K10. 聚类 使用颜色相近性来分割图片，来自Vorlesung。
img_dir = "test.jpg"
img_pil = Image.open(img_dir)
plt.imshow(img_pil)
plt.show()
testimg = np.array(img_pil)/255

# 获得整个图像的亮度
greyImg = 0.001+ np.sqrt(testimg[:, :,0]**2 + testimg[ : , :,1]**2 + testimg[ : , : ,2]**2)

# 让图片每个通道都除最大的亮度，哪个颜色通道上的最大就可以被选出
colorsonly = np.repeat(greyImg[ :, :, np.newaxis], 3, axis=2)

# 定义所需要的颜色
color = np.ones(testimg.shape)
color[:, :, 0] = 0.828
color[:, :, 1] = 0.509
color[:, :, 2] = 0.232

# 如果值距离所需颜色越相近，则值越大，即角度越大，注意sum在axis=2上
test = np.sum(color * colorsonly, axis=2)
test = test/ np.max(test)
test_angle = np.arccos(test)
test_angle[test_angle < 1.5] = 0

plt.imshow(test_angle, 'gray')
plt.show()
'''