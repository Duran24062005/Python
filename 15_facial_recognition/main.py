from deepface import DeepFace
import cv2 as cv

path = './assets/cara1.png'

faces = DeepFace.extract_faces(path)
# print(faces)

image = cv.imread(path)

area = faces[0]['facial_area']
# print(area)

x = area['x']
y = area['y']
width = area['w']
height = area['h']

# print(x,y,width,height)

cv.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
cv.imshow('face detected', image)
cv.waitKey(0)
