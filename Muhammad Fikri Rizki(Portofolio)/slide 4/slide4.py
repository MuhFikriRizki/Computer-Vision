### Muhammad Fikri Rizki
### D4 ELIN PENS
### Feature Detection, matching, object detection


import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv2.imread("poster.jpg") # queryImage
img1_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("object.jpg") # trainImage
img2_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# find the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_2,None)
kp2, des2 = sift.detectAndCompute(img2_2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel()
    h,w = img1_2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2_2 = cv2.polylines(img2_2,[np.int32(dst)],True,(0,255,0), 3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), singlePointColor = None, 
                    matchesMask = matchesMask, flags = 2)

img3 = cv2.drawMatches(img1_1,kp1,img2_1,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()












