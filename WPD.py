
import cv2
import imutils
import datetime

import plt

gun_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
FirstFrame= None
gun_exists= False

while True:
    ret,frame = camera.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if frame is None:
        break
    frame = imutils.resize(frame,width=500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
    if len(gun) >0:
        gun_exists = True
    for(x,y,w,h) in gun:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray= gray[y:y+h,x:x+w]
        roi_color= frame[y:y+h,x:x+w]
    if FirstFrame is None:
        FirstFrame = gray
        continue
    cv2.putText(frame,datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),(10,frame.shape[0] - 10),  cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    if gun_exists:
        print("SUCCESS!")
        plt.imshow(frame)
        break
    else:
        cv2.imshow(frame,"Nothing found")
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()


