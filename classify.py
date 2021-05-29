from sklearn.externals import joblib
from hog import HOG
import dataset
import cv2
import mahotas
import imutils
import numpy as np
from imutils.video import VideoStream
import time

model = joblib.load('data/svm.cpickle')

hog = HOG(orientations=18, pixelsPerCell= (10,10),
          cellsPerBlock=(1,1), transform=True)

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

#image = cv2.imread('1.png')
    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
    print(len(cnts))


    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)


        roi = gray[y:y+h , x:x+w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh>T]=255
        thresh =cv2.bitwise_not(thresh)

        thresh = dataset.deskew(thresh,20)
        thresh = dataset.center_extent(thresh,(20,20))

        cv2.imshow('thresh', thresh)

        hist = hog.describe(thresh)
        digit = model.predict([hist])[0]
        print('i think the number is {}',digit)

        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),1)
        cv2.putText(frame, str(digit), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0,255,0),2)

    cv2.imshow('finalimage', frame)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()