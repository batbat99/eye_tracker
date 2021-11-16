import cv2
import numpy as np

frontal_face = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('classifier/eye_pair_big.xml')
#eye_cascade = cv2.CascadeClassifier('classifier/eye_pair_small.xml')
eye_cascade = cv2.CascadeClassifier('classifier/haarcascade_eye.xml')

kernel = np.ones((4,4),np.uint8)
kernel2 = np.ones((8,8),np.uint8)

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = frontal_face.detectMultiScale(frame_gray)
    
    for face in faces:
        x,y,w,h = face
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        faceROI = frame_gray[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(faceROI, minNeighbors = 20)
        
        gaze_left = [False,False]
        gaze_right = [False,False]
        
        for i,(x2,y2,w2,h2) in enumerate(eyes):

            if i == 2:
                break

            
            
            
            frame = cv2.rectangle(frame, (x+x2,y+y2), (x+x2+h2,y+y2+w2), (255, 0, 0 ), 4)
            
            eyeROI = faceROI[y2:y2+w2,x2:x2+h2]
            
            
            scale_percent = 500 # percent of original size
            width = int(eyeROI.shape[1] * scale_percent / 100)
            height = int(eyeROI.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(eyeROI, dim, interpolation = cv2.INTER_AREA)
            resized = cv2.equalizeHist(resized)
            resized = cv2.GaussianBlur(resized,(5,5), cv2.BORDER_DEFAULT)
            
            rows, cols = resized.shape
            
            _, threshold = cv2.threshold(resized, 10, 255, cv2.THRESH_BINARY_INV)
            
            dilation = cv2.dilate(threshold,kernel,iterations = 4)
            erosion = cv2.erode(dilation,kernel,iterations = 2)
            
            #_, threshold2 = cv2.threshold(resized,40 , 255, cv2.THRESH_BINARY_INV)
            #erosion2 = cv2.erode(threshold2,kernel2,iterations = 3)
            #dilation2 = cv2.dilate(erosion2,kernel2,iterations = 1)
            
            contours,_ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            contours = contours[:1]
            #contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                    
            #contours2,_ = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #contours2 = sorted(contours2, key=lambda x: cv2.contourArea(x), reverse=True)
            #contours2 = contours2[:2]
            #contours2 = sorted(contours2, key=lambda ctr: cv2.boundingRect(ctr)[0])  
            
            #i = 0
            #while i<2:
            #    try:
            #        eye = contours2[i]
            #        pupil = contours[i]
            #        (xe, ye, we, he) = cv2.boundingRect(eye)
            #        (xp, yp, wp, hp) = cv2.boundingRect(pupil)
            #        cv2.rectangle(resized, (xe, ye), (xe + we, ye + he), (255, 0, 0), 2)
            #        cv2.line(resized, (xe + int(we/2), 0), (xe + int(we/2), rows), (0, 255, 0), 2)
            #        cv2.line(resized, (0, ye + int(he/2)), (cols, ye + int(he/2)), (0, 255, 0), 2)
            #        cv2.rectangle(resized, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
            #        cv2.line(resized, (xp + int(wp/2), 0), (xp + int(wp/2), rows), (0, 255, 0), 2)
            #        cv2.line(resized, (0, yp + int(hp/2)), (cols, yp + int(hp/2)), (0, 255, 0), 2)
            
            #        if (xp+(xp+wp)/2)-(xe+(xe+we)/2)>17:
            #            gaze_left[i] = True
            #        elif (xp+(xp+wp)/2)-(xe+(xe+we)/2)<-17:
            #            gaze_right[i] = True
            #    except:
            #        pass
                
            #    i+=1
            
            (xp, yp, wp, hp) = cv2.boundingRect(contours[0])
            cv2.rectangle(resized, (xp, yp), (xp + wp, yp + hp), (255, 0, 0), 2)
            cv2.line(resized, (xp + int(wp/2), 0), (xp + int(wp/2), rows), (0, 255, 0), 2)
            cv2.line(resized, (0, yp + int(hp/2)), (cols, yp + int(hp/2)), (0, 255, 0), 2)
            
            if (xp+int(wp/2))-(cols/2)>18:
                gaze_left[i] =True
            elif (xp+int(wp/2))-(cols/2)<-18:
                gaze_right[i] =True
            
            cv2.imshow("pupil"+str(i), erosion)
            cv2.imshow("eye"+str(i), resized)
            
         
            
        if gaze_left[0] and gaze_left[1]:
            print("looking left")
        elif gaze_right[0] and gaze_right[1]:
            print("looking right")
        else:
            print("looking center")



        
    
    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        