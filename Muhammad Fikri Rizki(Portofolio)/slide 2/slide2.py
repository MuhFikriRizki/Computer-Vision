### Muhammad Fikri Rizki
### D4 ELIN PENS
### Image Processing, filtering, select item


import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io, skimage.color as color, skimage.filters as filters, skimage.morphology as morphgy 


image = io.imread("dompet.jpeg", False)
gray_image = color.rgb2gray(image)
blurred_image = filters.gaussian(gray_image, sigma=2.0)
thresh = filters.threshold_mean(gray_image)

### create a histogram of the blurred grayscale image
histogram, bin_edges = np.histogram(blurred_image, bins=255, range=(0.0, 1.0))

#### create a mask based on the threshold
t = filters.threshold_otsu(blurred_image)
binaryMask = blurred_image < t
image_label_overlay = color.label2rgb(binaryMask, image=image, bg_label= 0)

fig, ax = plt.subplots(ncols=4, figsize=(20, 8))
ax[0].set_title('original')
ax[0].imshow(image, cmap='gray')
ax[1].set_title('grayScale')
ax[1].imshow(gray_image, cmap='gray')
ax[2].set_title(('binaryMask', t))
ax[2].imshow(binaryMask, cmap='gray')
ax[3].set_title('item selection')
ax[3].imshow(image_label_overlay, cmap='gray')
plt.tight_layout()
plt.show()