### Muhammad Fikri Rizki
### D4 ELIN PENS
### Qrcode, Barcode Scanner and Detection real time

import cv2
from pyzbar.pyzbar import _pixel_data, decode
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    _, img=cap.read()

    #### image processing
    capGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pixels, width, height = (_pixel_data(capGray))
    # print(pixels, '\n\n', width, '\n\n', height, '\n\n\n\n')
    capBlur1 = cv2.bilateralFilter(capGray, 7, 35, 35)
    edges1 = cv2.Canny(capBlur1, 75, 100, apertureSize=3)
    _, h, _ = img.shape

    for i in range (h):
        bg = cv2.rectangle(img, (0,0), (i,h), (0,0,0), 2)

    contours,_=cv2.findContours(edges1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,255), 1)


    #### detect QRcode & Barcode
    code = decode(capGray)     
    # print(code)
    '''[Decoded(data=b'CKDCA33B3E2Q2A2ZC7ZA', 
                type='QRCODE', 
                rect=Rect(left=233,top=190, width=88, height=91), 
                polygon=[Point(x=233, y=196), 
                Point(x=235, y=281), Point(x=321, y=278), Point(x=318, y=190)], 
                quality=1, orientation='UP')]'''

    #### create decode result and polygon of code
    for array in code:
        # print(len(array.polygon))
        # print(array.polygon[0].x)
        teks = array.data.decode('utf-8')
        n = 5
        m = 7
        pointPoly = []
        for i in range (len(array.polygon)):
            var = [array.polygon[i].x, array.polygon[i].y]
            # print(var)
            pointPoly.append(var)
            pts = np.array(pointPoly, np.int32)
            pts = pts.reshape((-1,1,2))

            if i==0:
                a = array.polygon[i].x
                b = array.polygon[i].y

            polygon = cv2.polylines(img, [pts], True, (0,255,0), 2)
            text = cv2.putText(img, teks, (a-n, b-m),
                                 cv2.FONT_HERSHEY_COMPLEX, 
                                 0.5, (255,0,0), 1, cv2.LINE_AA)

        # print(pts)

    
    # print(img.shape)
    cv2.imshow('result', img)
    cv2.waitKey(1)