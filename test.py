import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import urllib.request
import numpy as np
import time
model=YOLO('yolov8s.pt')





url = 'http://192.168.0.104/800x600.jpg'


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)




while True:    
    imgResp = urllib.request.urlopen(url)
#    count += 1
#    if count % 3 != 0:
#        continue
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    image = cv2.imdecode(imgNp, -1)
#    frame=cv2.resize(image,(1020,500))
   

    results=model.predict(image)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list=[]
    list1=[]         
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cvzone.putTextRect(image,f'{c}',(x1,y2),1,1)   
    cv2.imshow("IMAGE", image)
    if cv2.waitKey(1)&0xFF==27:
        break
cv2.destroyAllWindows()


