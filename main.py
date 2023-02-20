import cv2
# import os
import HandTrackingModule as htm
import numpy as np
import time

brushsize = 15
erasersize=50

drawColor=(0,0,255)

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
imgcanvas=np.zeros((720,1280,3),np.uint8)


detector=htm.handDetector(detectionCon=0.85)
pTime = 0

while True:

#1.  import image
    success,img = cap.read()
    img=cv2.flip(img,1)
    cv2.rectangle(img,(10,0),(1270,110),(0,0,0),cv2.FILLED)

    cv2.rectangle(img,(20,10),(210,100),(0,0,255),cv2.FILLED)
    cv2.rectangle(img, (230, 10), (450, 100), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (470, 10), (680, 100), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(img,(700,10),(920,100),(0,255,255),cv2.FILLED)
    cv2.rectangle(img, (940, 10), (1260, 100), (255, 255, 255),cv2.FILLED)
    cv2.putText(img,'ERASER',(1050,75),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)





#2. find hand landmarks
    img=detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)

    if len(lmlist)!=0:
        # print(lmlist)

        #tip of 2 fingers
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

#3. check which finger is up
        fingers=detector.fingersUp()
        # print(fingers)

#4.if selection mode - two finger is up
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            print('selection Mode')

            #checking for the click
            if y1 < 130:
                if 20 < x1 < 210:

                    drawColor = (0, 0, 255)
                elif 230 < x1 < 450:

                    drawColor = (0, 255, 0)
                elif 460 < x1 < 680:

                    drawColor = (255, 0, 0)
                elif 700 < x1 < 920:

                    drawColor = (0,255,255)
                elif 940<x1<1260:

                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 + 15), (x2, y2 + 15), drawColor, cv2.FILLED)

#5. if drawing mode - index finger is up

        if (fingers[1] and not fingers[2]):
                     
            cv2.putText(img, "Writing Mode", (900,680), fontFace= cv2.FONT_HERSHEY_COMPLEX, color= (255,255,0), thickness=2, fontScale=1)
            cv2.circle(img, (x1,y1),15, drawColor, thickness=-1)

            if xp ==0 and yp ==0:
                xp =x1 
                yp =y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1),color= drawColor, thickness=erasersize)
                cv2.line(imgcanvas, (xp,yp),(x1,y1),color= drawColor, thickness=erasersize)

            else:
                cv2.line(img, (xp,yp),(x1,y1),color= drawColor, thickness=brushsize)
                cv2.line(imgcanvas, (xp,yp),(x1,y1),color= drawColor, thickness=brushsize)
            
            xp , yp = x1, y1

    imgGray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgcanvas)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 200), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)


    img=cv2.addWeighted(img,1 ,imgcanvas,0.5,0)
    cv2.imshow('image',img)

    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
        break


