# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:47:52 2020

@author: Snehangsu
"""

import numpy as np
import cv2
import math as m


hand_cascade = cv2.CascadeClassifier('C:/Users/babai/Desktop/haarcascade/aGest.xml')


cap = cv2.VideoCapture(0)
#fps = cap.get(cv2.CAP_PROP_FPS)
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
out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hand = hand_cascade.detectMultiScale(gray,1.3,5)
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
    for (x,y,w,h) in hand:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        d=((110*f)/w)/1000
        hi=d*m.sin(m.radians(p_deg*abs((360-(y+(h/2))))))
        tempStr="Distance= "+str(d)+" , Height = X + "+str(hi)
        cv2.putText(img,tempStr,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,0),5,1)
        
        if(c==1):    
          d2=d
          x2=x
          y2=y
          w2=w
          h2=h
          s=m.sqrt(np.square(d2*m.sin(m.radians(pv_deg*abs((640-(x2+(w2/2))))))-d1*m.sin(m.radians(pv_deg*abs((640-(x1+(w1/2)))))))+np.square(d2*m.sin(m.radians(p_deg*abs((360-(y2+(h2/2))))))-d1*m.sin(m.radians(p_deg*abs((360-(y1+(h1/2)))))))+np.square(d2*m.cos(m.radians(p_deg*abs((360-(y2+(h2/2))))))-d1*m.cos(m.radians(p_deg*abs((360-(y1+(h1/2))))))))/((c2+1)/30)
          tempStr2="Speed = "+str(s)
          cv2.putText(img,tempStr2,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,0),5,1)
          print(c)
          c=0
          c2=0
        if(c==0):
          d1=d
          x1=x 
          y1=y
          w1=w
          h1=h
          c+=1
    
    cv2.imshow('img',img)
    out.write(img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()