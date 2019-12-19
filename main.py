# -*- coding: utf-8 -*-

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
from skimage import transform

img = cv2.imread('multi.jpg')

face_detection = cv2.CascadeClassifier('haarcascade_face.xml')
clf = load_model('model.h5')
emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']


def load_img(image):
    # np_img = Image.open(filename)
    np_img = np.array(image).astype('float32')
    np_img = transform.resize(np_img, (48,48,3))
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray, 1.3, 3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        image = gray[y:y+h, x:x+w]
        # cv2.imwrite('output.jpg', image)
        np_img = load_img(image)
        
        preds = clf.predict(np_img)[0]
        emotion = emotions[np.argmax(preds)]
        prob = np.max(preds)

        cv2.putText(frame, "{}: {:.2f}%".format(emotion, prob*100),
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)

        # print(emotion, prob)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


