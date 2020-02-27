import cv2
import sys
from matplotlib import pyplot as plt

imagepath = r'OpenCv/image0.jpg'
dir_cascade_file = r'OpenCv/opencv/haarcascades_cuda/'
cascade_file = dir_cascade_file + "haarcascade_frontalface_alt.xml"
class_cascade = cv2.CascadeClassifier(cascade_file)
imageBGR = cv2.imread(imagepath)
plt.imshow(imageBGR[:,:,::-1])
plt.axis('off')
plt.show()

image_gray = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)

plt.imshow(image_gray)
plt.axis('off')
plt.show()

faces = class_cascade.detectMultiScale(
    image_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE,
)

print('il y a {0} de visages'.format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(imageBGR, (x, y), (w + x, y + h), (0, 255, 0), 2)

plt.imshow(imageBGR[:,:,::-1])
plt.axis('off')
plt.show()