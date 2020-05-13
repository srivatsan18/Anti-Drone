# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:06:42 2020

@author: babai
"""
import math as m
import numpy as np
import cv2
#tracker = cv2.TrackerKCF_create()
#tracker = cv2.TrackerMIL_create()
tracker = cv2.TrackerTLD_create()
#tracker = cv2.TrackerMedianFlow_create()
#tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerMOSSE_create()
#cap = cv2.VideoCapture('1003.jpg')
out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
def calc_roi():
        cap=cv2.VideoCapture('C://Users//babai//Downloads//test_drone1.mp4')
        success, frame = cap.read()
        bbox = cv2.selectROI("Tracking",frame, False)
        return frame,bbox;
    
frame,roi=calc_roi()
tracker.init(frame,roi)
cap=cv2.VideoCapture('C://Users//babai//Downloads//test_drone1.mp4')
t=m.atan(m.radians(39.3))
f=np.sqrt(np.square(1280)+np.square(720))*t
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

h_fov=2*m.degrees(m.atan(640/f))
v_fov=2*m.degrees(m.atan(360/f))
p_deg=h_fov/720
pv_deg=v_fov/1280
change_res(1280,720)
c=0
c2=0

#success, frame = cap.read()
def drawBox(img,bbox,c2):
      x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
      global c
     
      global d1
      global x1
      global y1
      global w1
      global h1
      global d2
      global x2
      global y2
      global w2
      global h2
      cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
      cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      d=round((((110*f)/w)/1000),4)
      hi=round(d*m.sin(m.radians(p_deg*abs((360-(y+(h/2)))))),4)
      tempStr="Distance= "+str(d)+" , Height = X + "+str(hi)
      cv2.putText(img,tempStr,(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(200,180,0),5,1)
        
      if(c==1):    
         d2=d
         x2=x
         y2=y
         w2=w
         h2=h
         s=round((m.sqrt(np.square(d2*m.sin(m.radians(pv_deg*abs((640-(x2+(w2/2))))))-d1*m.sin(m.radians(pv_deg*abs((640-(x1+(w1/2)))))))+np.square(d2*m.sin(m.radians(p_deg*abs((360-(y2+(h2/2))))))-d1*m.sin(m.radians(p_deg*abs((360-(y1+(h1/2)))))))+np.square(d2*m.cos(m.radians(p_deg*abs((360-(y2+(h2/2))))))-d1*m.cos(m.radians(p_deg*abs((360-(y1+(h1/2))))))))/((c2+1)/30)),4)
         tempStr2="Speed = "+str(s)
         cv2.putText(img,tempStr2,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),5,1)
         
         c=0
         c2=0
      if(c==0):
          d1=d
          x1=x 
          y1=y
          w1=w
          h1=h
          c+=1

while True:
    d1=0
    x1=0
    y1=0
    w1=0
    h1=0
    d2=0
    x2=0
    y2=0
    w2=0
    h2=0
    c2+=1
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)

    if success:
        drawBox(img,bbox,c2)
    else:
        cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(img,(15,15),(200,90),(255,0,255),2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2);
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if fps>60: myColor = (20,230,20)
    elif fps>20: myColor = (230,20,20)
    else: myColor = (20,20,230)
    cv2.putText(img,str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv2.imshow("Tracking", img)
    out.write(img)
    if cv2.waitKey(1) & 0xff == ord('q'):
       break
   
cap.release()
out.release()
cv2.destroyAllWindows()