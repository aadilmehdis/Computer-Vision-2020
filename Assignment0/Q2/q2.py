import cv2
import numpy as np

cap = cv2.VideoCapture(0)

i=0
while(True):
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('Video Camera Feed'.format(i), frame)
        cv2.imwrite(filename='{}.jpg'.format(i), img=frame)
        i+=1
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()