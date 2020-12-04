
import numpy as np
from keras.models import model_from_json
import cv2
import os,sys
import operator


##loaded model

json_file = open("model-bw.json","r")

model_json=json_file.read()
json_file.close()

loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")

print("loaded model from disk")

categories = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five"}

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    x1 = int(0.5*frame.shape[1])
    y1= 10
    
    x2 = frame.shape[1]-10
    y2=int(0.5*frame.shape[1])
    
    cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(0,0,255),1)
    
    roi = frame[y1:y2,x1:x2]
    
    roi = cv2.resize(roi,(64,64))
    roi  =cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,test_img = cv2.threshold(roi,120,255,cv2.THRESH_BINARY)
    
    cv2.imshow("test", test_img)
    
    
    result = loaded_model.predict(test_img.reshape(1,64,64,1))
    
    prediction ={"zero":result[0][0],
                 "one":result[0][1],
                 "two":result[0][2],
                 "three":result[0][3],
                 "four":result[0][4],
                 "five":result[0][5]
                 }
    
    prediction = sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)
    
    cv2.putText(frame,prediction[0][0],(10,120),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    cv2.imshow("frame",frame)
    
    k =cv2.waitKey(10) & 0xFF
    
    if k==27:
        break

cap.release()

cv2.destroyAllWindows()
    
    

