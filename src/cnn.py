import numpy as np
import os
import cv2
import tensorflow as tf
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.models import Sequential
from math import ceil
import matplotlib.pyplot as plt
from keras.src.utils import img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from keras.src.saving import load_model
import datetime as dt

class HandClassifierModel:
  
  def __init__(self):
    self.model = None
    
  def load_data(self, data_dir):
    self.data_dir = data_dir
    
    images = []
    labels = []
    folders = sorted(os.listdir(self.data_dir))
    
    for i, folder in enumerate(folders):
      
      print(f"Loading images from folder {folder}, has started.")
      for image in os.listdir(f"{self.data_dir}/{folder}"):

        img = cv2.imread(f"{self.data_dir}/{folder}/{image}", 0)
        img = self.preprocess_image(img)

        images.append(img)
        labels.append(i)

      break

    images = np.array(images)
    
    X = images.astype('float32')/255.0
    y = to_categorical(labels)
    
    return X, y
  
  def preprocess_image(self, image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(res, (64, 64))
    img = img_to_array(img)
    return img
  
  def build_model(self):
    self.model = Sequential()
    self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Conv2D(256, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    
    self.model.add(Conv2D(512, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.5))

    self.model.add(Conv2D(512, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.5))

    self.model.add(Flatten())

    self.model.add(Dropout(0.5))
    
    self.model.add(Dense(1024, activation='relu'))
    
    self.model.add(Dense(1, activation='softmax'))
    
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
  def train(self, X_train, y_train, epochs=5, batch_size=64):
    history = self.model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    self.history_attribute = history.history
  
  def evaluate(self, X_test, y_test):
    test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
    print(f"Loss en datos de prueba: {test_loss}")
    print(f"Precisión en datos de prueba: {test_accuracy}")
    return test_loss, test_accuracy
  
  def plot_training(self):
    plt.figure(figsize=(8,5))
    plt.plot(self.history_attribute['accuracy'], label='train_accuracy',)
    plt.plot(self.history_attribute['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("HandsClassifierModel")

    plt.show()
    
  def save_model(self, filename):
    model.save_model(filename)
  
  def load_trained_model(self, model_path):
    self.model = load_model(model_path)
    
  def predict_gesture(self, image):
    img = self.preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    
    prediction = self.model.predict(img)
    confidence = np.max(prediction)
    
    predicted_class = np.argmax(prediction)
    
    gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    
    predicted_gesture = gesture_classes[predicted_class]

    return predicted_gesture, confidence
    
    
if __name__ == "__main__":
  DATA_TRAIN_DIR = '../data/asl_alphabet_train/asl_alphabet_train'
  model = HandClassifierModel()
  X, y = model.load_data(DATA_TRAIN_DIR)
  
  X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=89
  )
  model.build_model()
  model.train(X_train, y_train)
  model.plot_training()
  
  current_date_and_time = dt.datetime.now().strftime("%Y%m%d%H")
  SAVING_PATH = f"../models/modelo_{current_date_and_time}.h5"
  model.save_model(SAVING_PATH)
  
  DATA_TEST_DIR = '../data/asl_alphabet_test/asl_alphabet_test'
  X, y = model.load_data(DATA_TEST_DIR)
  model.evaluate(X_test, y_test)