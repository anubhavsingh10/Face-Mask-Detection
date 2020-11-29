import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import random
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img



model = tf.keras.models.load_model(r"C:\Users\hschahar\Desktop\GUIDED_PROJECTS\Face Mask Detection\FaceDetectionModel.h5")

facedetector =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0,cv2.CAP_DSHOW)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

while True:
  ret,img = source.read()
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face = facedetector.detectMultiScale(gray,1.1,6)

  for (x,y,w,h) in face:
    
        face_img=gray[y:y+w,x:x+w]

        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
       	
       	face_img=cv2.resize(face_img,(100,100))
       	
       	face_img=np.reshape(face_img,(1,100,100,1))
       	

        result=model.predict(face_img)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
  cv2.imshow('LIVE',img)
  key=cv2.waitKey(1)
    
  if(key==27):
    break
        
cv2.destroyAllWindows()
source.release()