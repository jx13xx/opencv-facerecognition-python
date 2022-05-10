
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces =  haar.detectMultiScale(gray,1.3,5)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,255), 2)

    return img

def detect_face_video():

    cap = cv2.VideoCapture('video.mp4')

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = face_detect(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('object_detect',frame)
        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
def detect_face():
    # Read the image file
    img = cv2.imread('female_1.png')

    # convert the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # apply haar cascade
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = haar.detectMultiScale(gray,1.3,5)
    cv2.rectangle(img,(13, 13), (13+261,13+261), (0,255,0),2)
    plt.imshow(img)
    plt.show()

    print(faces)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    img = cv2.imread('female_1.png')
    img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray,cmap='gray')
    plt.show()
    # Press the green button in the gutte`r to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # detect_face()
    detect_face_video()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
