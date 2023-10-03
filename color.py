import matplotlib.pyplot as plt 
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io

img = skimage.io.imread('peppers.png')
img = skimage.img_as_float(img)

sigma = 0.15
imgn = random_noise(img, var=sigma**2)
sigma_est = estimate_sigma(imgn, channel_axis= None, average_sigmas=True)

img_visushrink = denoise_wavelet(imgn, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5, wavelet='coif5', channel_axis= None, convert2ycbcr=True, rescale_sigma=True)

psnr_noisy = peak_signal_noise_ratio(img, imgn)
psnr_visu = peak_signal_noise_ratio(img, img_visushrink)

plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title('Original Image', fontsize=30)

plt.subplot(2,2,2)
plt.imshow(imgn, cmap=plt.cm.gray)
plt.title('Noisy Image', fontsize=30)

plt.subplot(2,2,3)
plt.imshow(img_visushrink, cmap=plt.cm.gray)
plt.title('Clear Image', fontsize=30)

plt.show()

print('PSNR [Original Vs. Noisy Image]:', psnr_noisy)
print('PSNR [Original Vs. Denoised]:', psnr_noisy)
wavelets()