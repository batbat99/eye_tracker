from typing import List
import dlib
import cv2 
import numpy as np

from transform import four_point_transform

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

frontal_face = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')

kernel = np.ones((4,4),np.uint8)

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = frontal_face.detectMultiScale(frame_gray, minNeighbors=10)
    
    for face in faces:
        (x,y,w,h) = face
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        faceROI = frame_gray[y:y+h,x:x+w]
        rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        shape  = predictor(frame_gray,rect)

        eyes = [[],[]]

        for b in range(4):
            x = shape.part(b).x
            y = shape.part(b).y
            i = b//2
            eyes[i].append([x,y])

            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(frame, (shape.part(4).x, shape.part(4).y), 1, (0, 0, 255), -1)

        for e,eye in enumerate(eyes):
            height = abs(eye[1][0]-eye[0][0])
            width = abs(eye[1][0]-eye[0][0])
            bbox = [(eye[0][0],eye[0][1]-height/2),(eye[1][0],eye[1][1]-height/2),(eye[1][0],eye[1][1]+height/2),(eye[0][0],eye[0][1]+height/2)]
            bbox = np.array([bbox],np.int32)
            frame = cv2.polylines(frame, bbox, True, (255,120,255),1)
            eye_frame = four_point_transform(frame_gray,bbox[0])

            scale_percent = 500 # percent of original size
            width = int(eye_frame.shape[1] * scale_percent / 100)
            height = int(eye_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            eye_frame = cv2.resize(eye_frame, dim, interpolation = cv2.INTER_AREA)
            eye_frame = cv2.equalizeHist(eye_frame)
            eye_frame = cv2.GaussianBlur(eye_frame,(5,5), cv2.BORDER_DEFAULT)

            rows, cols = eye_frame.shape
            _, threshold = cv2.threshold(eye_frame, 10, 255, cv2.THRESH_BINARY_INV)
            dilation = cv2.dilate(threshold,kernel,iterations = 5)
            erosion = cv2.erode(dilation,kernel,iterations = 2)

            contours,_ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            contours = contours[:1]

            (xp, yp, wp, hp) = cv2.boundingRect(contours[0])
            cv2.rectangle(eye_frame, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
            cv2.line(eye_frame, (xp + int(wp/2), 0), (xp + int(wp/2), rows), (0, 255, 0), 2)
            cv2.line(eye_frame, (0, yp + int(hp/2)), (cols, yp + int(hp/2)), (0, 255, 0), 2)
            
            cv2.imshow(str(e+1)+" pupil", dilation)
            cv2.imshow(str(e+1)+" eye", eye_frame)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        