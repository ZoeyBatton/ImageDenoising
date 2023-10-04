import matplotlib.pyplot as plt 
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
import cv2
import numpy as np

img = skimage.io.imread('C:/Users/hoodi/OneDrive/Desktop/New folder/OIP.png')
img = skimage.img_as_float(img)

sigma = 0.1
imgn = random_noise(img, var=sigma**2)
sigma_est = estimate_sigma(imgn, average_sigmas=True)

img_bayes = denoise_wavelet(imgn, method='BayesShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5, wavelet='bior6.8', rescale_sigma=True)
img_visushrink = denoise_wavelet(imgn, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5, wavelet='bior6.8', rescale_sigma=True)

psnr_noisy = peak_signal_noise_ratio(img, imgn)
psnr_bayes = peak_signal_noise_ratio(img, img_bayes)
psnr_visu = peak_signal_noise_ratio(img, img_visushrink)

plt.figure(figsize=(30,30))
# plt.subplot(2,2,1)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.title('Original Image', fontsize=30)

plt.subplot(2,3,1)
plt.imshow(imgn, cmap=plt.cm.gray)
plt.title('Noisy Image', fontsize=25)

plt.subplot(2,3,2)
plt.imshow(img_bayes, cmap=plt.cm.gray)
plt.title('Denoised Image- Bayes', fontsize=25)

directory = 'C:/Users/hoodi/OneDrive/Desktop/New folder/'
img1 = img_bayes
filename = '1.png'
cv2.imwrite(filename, img1)

img2 = img_visushrink
filename = '2.png'
cv2.imwrite(filename, img2)

plt.subplot(2,3,3)
plt.imshow(img_visushrink, cmap=plt.cm.gray)
plt.title('Denoised Image- Visu', fontsize=25)



print('PSNR [Original Vs. Noisy Image]:', psnr_noisy)
print('PSNR [Original Vs. Denoised]:', psnr_noisy)

import cv2

images = cv2.imread('C:/Users/hoodi/OneDrive/Desktop/New folder/OIP.png')    

gray = cv2.imread ('C:/Users/hoodi/OneDrive/Desktop/New folder/OIP.png', 0 )
th, threshed = cv2.threshold (gray, 100 , 255 ,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cnts = cv2.findContours (threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) [ - 2 ]

plt.subplot(2,3,4)
plt.imshow(threshed, cmap=plt.cm.gray)

cv2.waitKey()

images = cv2.imread('C:/Users/hoodi/OneDrive/Desktop/New folder/1.png')    

gray = cv2.imread ('C:/Users/hoodi/OneDrive/Desktop/New folder/1.png', 0 )
th, threshed = cv2.threshold (gray, 100 , 255 ,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cnts = cv2.findContours (threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) [ - 2 ]

plt.subplot(2,3,5)
plt.imshow(threshed, cmap=plt.cm.gray)

cv2.waitKey()

images = cv2.imread('C:/Users/hoodi/OneDrive/Desktop/New folder/2.png')    

gray = cv2.imread ('C:/Users/hoodi/OneDrive/Desktop/New folder/2.png', 0 )
th, threshed = cv2.threshold (gray, 100 , 255 ,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cnts = cv2.findContours (threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) [ - 2 ]

plt.subplot(2,3,6)
plt.imshow(threshed, cmap=plt.cm.gray)

cv2.waitKey()


plt.show()
# imga = cv2.imread(img_bayes)
# grays = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
# blur = cv2.blur(grays,(71,71))

# diff = cv2.subtract(blur, grays)
# ret, th = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY_INV)
# plt.subplot(2,3,4)
# plt.imshow("threshold", th, cmap=plt.cm.gray)
# cv2.waitKey(0)