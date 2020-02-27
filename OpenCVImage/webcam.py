import numpy as np
import cv2

dir_cascade_file = r'OpenCv/opencv/haarcascades_cuda/'
cascade_file = dir_cascade_file + "haarcascade_frontalface_default.xml"
class_cascade = cv2.CascadeClassifier(cascade_file)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = class_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE,
    # )
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (w + x, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()