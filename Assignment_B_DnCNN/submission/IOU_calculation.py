from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

mask_pic = Image.open('./sheep_seg_from_givenMask.png')
sheep_seg = Image.open('./sheep_segmentation.png')

mask_array = np.array(mask_pic)
sheep_array = np.array(sheep_seg)

rM_channel, gM_channel, bM_channel = cv2.split(mask_array)
rS_channel, gS_channel, bS_channel = cv2.split(sheep_array)

rM_192_ind = rM_channel == 128 # mask_array sheep segmentation part
rS_192_ind = rS_channel == 128 # model_trained sheep segmentation part

ind_intersection = np.all([rM_192_ind,rS_192_ind],axis = 0 ) #intersection,  And logic operation
intersection_value = ind_intersection.sum()


inter_image = np.zeros_like(rM_channel).astype(np.uint8)

inter_image[ind_intersection] = 255


inter_image = Image.fromarray(inter_image)
plt.imshow(inter_image, cmap = 'gray')
plt.title(label = 'Intersection')
plt.show()

Union_array = np.zeros_like(rM_channel).astype(np.uint8)
Union_array[rM_192_ind] = 1
Union_array[rS_192_ind] = 1

union_image = np.zeros_like(rM_channel).astype(np.uint8)
union_image = Image.fromarray(Union_array)
plt.imshow(union_image, cmap = 'gray')
plt.title(label = 'Union')
plt.show()

Uniot_value = Union_array.sum()

IOU_rate = intersection_value / Uniot_value
print('The IOU rate is: ', IOU_rate)

