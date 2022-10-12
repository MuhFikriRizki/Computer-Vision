### Muhammad Fikri Rizki
### D4 ELIN PENS
### Image Processing, filtering, edge detection


import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import feature
import skimage.io as io, skimage.color as color

imageAwal = io.imread("bacteria.jpg", False)[50:-5, 50:-160]
image = color.rgb2gray(imageAwal)
image = random_noise(image, mode='speckle', mean=0.8)

# Compute the Canny filter for two values of sigma
nsigma = 1.2
edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=nsigma)

image_label_overlay = color.label2rgb(edges1, image=image, bg_label= 0)

# display results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

ax[0].imshow(imageAwal, cmap='gray')
ax[0].set_title('original image', fontsize=16)

ax[1].imshow(edges1, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma=1.2$', fontsize=16)

ax[2].set_title('selection', fontsize=16)
ax[2].imshow(image_label_overlay, cmap='gray')


fig.tight_layout()
plt.show()