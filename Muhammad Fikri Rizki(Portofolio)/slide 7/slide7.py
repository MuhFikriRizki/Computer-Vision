import cv2
import cvzone as cz
from cvzone.ColorModule import ColorFinder as cf


mycf  = cf(False)

color = {'hmin': 32, 'smin': 96, 'vmin': 75, 'hmax': 49, 'smax': 255, 'vmax': 255}

cap = cv2.VideoCapture('vid1.mp4')

while 1:
    _, vid = cap.read()
    imgNew, mask = mycf.update(vid, color)
    imgContour, contour = cz.findContours(vid, mask, minArea=300)

    if contour:
        x, y = contour[0]['center']
        # print(x,y)
        
        vid = cv2.circle(vid, (x,y), 0, (255,0,0), 30, cv2.FILLED)

    cv2.imshow('img tracking', imgContour)
    # cv2.imshow('find color', imgNew)
    cv2.imshow('img', vid)
    cv2.waitKey(50)
