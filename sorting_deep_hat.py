#!/usr/bin/env python3

from tensorflow.keras import models
import cv2
import numpy as np

class sorting_deep_hat:
    def __init__(self, model_path):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = models.load_model(model_path)    


    def estimate(self, input_image_path):
        self.image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,\
                                                    scaleFactor= 1.11,\
                                                    minNeighbors= 3,\
                                                    #minSize=(70, 70)\
                                                    )

        self.result_data = []
        for (x, y, w, h) in faces:
            
            #既に検出された顔領域内に顔が検出された場合は除外
            for (xx, yy, ww, hh) in self.result_data:
                if xx < x and x < xx + ww:
                    if yy < y and y < yy + hh:
                        continue
                if xx < x + w and x + w < xx + ww:
                    if yy < y + h and y + h < yy + hh:
                        continue

            face_image = self.image[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (100, 100))
    
            b,g,r = cv2.split(face_image)
            in_data = cv2.merge([r,g,b])
            in_data = np.array([in_data / 255.])

            index = np.argmax(self.model.predict(in_data))

            if index == 0:
                house_name = 'Glyffindor'
            elif index == 1:
                house_name = 'Hufflpuff'
            elif index == 2:
                house_name = 'Ravenclaw'
            elif index == 3:
                house_name = 'Slytherin'

            self.result_data.append([x, y, w, h, house_name])
    
    def draw(self, output_image_path):
        for (x, y, w, h, house_name) in self.result_data:
            color = ()
            if house_name == 'Glyffindor':
                color = (0, 0, 255)
            elif house_name == 'Hufflpuff':
                color = (0, 255, 255)
            elif house_name == 'Ravenclaw':
                color = (255, 0, 0)
            elif house_name == 'Slytherin':
                color = (0, 255, 0)

            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.image, house_name, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, color, 4)
            cv2.imwrite(output_image_path, self.image)
        
if __name__ == '__main__':
    model_path = 'models/sorting_deep_hat.h5'
    input_image_path = 'data/sample/harrypotter.jpg'
    output_image_path = 'output.jpg'

    sdt = sorting_deep_hat(model_path)
    sdt.estimate(input_image_path)
    print(sdt.result_data)
    sdt.draw(output_image_path)
