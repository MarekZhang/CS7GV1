import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import os, time, datetime

noise_array = np.load('noise.npy')
noise_array = noise_array[0][0]
print(noise_array)
print(noise_array.min())
print(noise_array.max())

noise_array = noise_array / noise_array.max() *255
noise_array = noise_array + 127

ind_neg = noise_array < 0
ind_plu = noise_array > 255

noise_array[ind_neg] = 0
noise_array[ind_plu] = 255

print('mean is:' ,noise_array.mean())
print(noise_array.min())
print(noise_array.max())
print('Standaerd deviation is: ', noise_array.std())

path = './np.png' 

imsave(path, np.uint8(np.clip(noise_array*255.0, 0, 255)))


