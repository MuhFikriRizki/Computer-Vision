### Muhammad Fikri Rizki
### D4 ELIN PENS
### Contour Detection real time


import cv2

cam = cv2.VideoCapture(0)
while True:
    _, capture=cam.read()
    
    capGray = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY)
    capBlur1 = cv2.bilateralFilter(capGray, 7, 35, 35)
    edges1 = cv2.Canny(capBlur1, 60, 150, apertureSize=3)

    contours,_=cv2.findContours(edges1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(capture, contours, -1, (0,255,0), 1)

    cv2.imshow('deteksi kontur', capture)
    cv2.waitKey(1)



