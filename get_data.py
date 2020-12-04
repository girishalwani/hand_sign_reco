import cv2
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")

model = "train"
directory = "data/"+model+"/"

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    frame=cv2.flip(frame,1)

    count={
        "zero": len(os.listdir(directory+"/0")),
        "one" : len(os.listdir(directory+"/1")),
        "two" : len(os.listdir(directory+"/2")),
        "three" : len(os.listdir(directory+"/3")),
        "four" : len(os.listdir(directory+"/4")),
        "five" : len(os.listdir(directory+"/5"))
    }

    cv2.putText(frame,"Mode : "+model,(10,50),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),1)
    cv2.putText(frame,"Image Count : ",(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"Zero : "+str(count["zero"]),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"one : "+str(count["one"]),(10,140),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"two : "+str(count["two"]),(10,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"three : "+str(count["three"]),(10,180),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"four : "+str(count["four"]),(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    cv2.putText(frame,"five : "+str(count["five"]),(10,220),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)


    x1=int(0.5*frame.shape[1])
    y1 = 10

    x2=frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(255,0,0),1)

    roi = frame[y1:y2,x1:x2]
    roi=cv2.resize(roi,(64,64))

    cv2.imshow("frame",frame)

    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,roi = cv2.threshold(roi,120,255,cv2.THRESH_BINARY)
    cv2.imshow("ROI",roi)

    k = cv2.waitKey(10) & 0xFF

    if k==27:
        break
    if k==ord('0'):
        cv2.imwrite(directory+"0/"+str(count['zero'])+'.jpg',roi)
    if k==ord('1'):
        cv2.imwrite(directory+"1/"+str(count['one'])+'.jpg',roi)
    if k==ord('2'):
        cv2.imwrite(directory+"2/"+str(count['two'])+'.jpg',roi)
    if k==ord('3'):
        cv2.imwrite(directory+"3/"+str(count['three'])+'.jpg',roi)
    if k==ord('4'):
        cv2.imwrite(directory+"4/"+str(count['four'])+'.jpg',roi)
    if k==ord('5'):
        cv2.imwrite(directory+"5/"+str(count['five'])+'.jpg',roi)

cap.release()
cv2.destroyAllWindows()
