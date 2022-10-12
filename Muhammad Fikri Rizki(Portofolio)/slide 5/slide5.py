### Muhammad Fikri Rizki
### D4 ELIN PENS
### Face Detection


import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
while True:
	_, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# draw a rectangle and text in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		cv2.putText(img, 'wajah', (x,y-7),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1, cv2.LINE_AA)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Detects eyes of different sizes in the input image
		eyes = eye_cascade.detectMultiScale(roi_gray)

		#To draw a rectangle and text in eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
			cv2.putText(roi_color, 'mata', (ex,ey-4),cv2.FONT_HERSHEY_PLAIN, 1, (0,127,255), 1, cv2.LINE_8)

	
	cv2.imshow('img',img)
	k = cv2.waitKey(1)
