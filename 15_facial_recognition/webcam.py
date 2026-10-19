from deepface import DeepFace
import cv2

path = './assets/capture.jpg'

camera = cv2.VideoCapture(0)
camera.read()
camera.read()
camera.read()
success, photo = camera.read()

camera.release()

if success:
    cv2.imwrite('./assets/capture2.jpg', photo)
    resutl = DeepFace.verify(img1_path=photo, img2_path=path)
    # print(resutl)
    if resutl['verified']:
        text = "Alexi"
        color = (0, 255, 0)
    else:
        text = "Unknown"
        color = (0, 0, 255)

    cv2.putText(photo, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow('photo', photo)
    cv2.waitKey(0)
