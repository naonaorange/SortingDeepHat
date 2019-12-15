#!/usr/bin/env python3

from tensorflow.keras import models
import cv2
import numpy as np

class sorting_deep_hat:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def estimate(self, input_image_path, model_path):
        image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,\
                                                    scaleFactor= 1.1,\
                                                    minNeighbors= 3,\
                                                    minSize=(70, 70))

        ret = []
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (100, 100))
    
            b,g,r = cv2.split(face_image)
            in_data = cv2.merge([r,g,b])
            in_data = np.array([in_data / 255.])

            model = models.load_model(model_path)    
            result = np.argmax(model.predict(in_data))

            if result == 0:
                house_name = 'Glyffindor'
            elif result == 1:
                house_name = 'Hufflpuff'
            elif result == 2:
                house_name = 'Ravenclaw'
            elif result == 3:
                house_name = 'Slytherin'

            ret.append([x, y, w, h, house_name])
        return ret
        
if __name__ == '__main__':
    model_path = 'models/sorting_deep_hat.h5'
    input_image_path = 'data/sample/harrypotter.jpg'
    output_image_path = 'output.jpg'
    face_rects = ()
    houses = []

    sdt = sorting_deep_hat()
    sdt.estimate(input_image_path, model_path)
