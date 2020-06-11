"""Test script for plant Contamination classification"""

from config import cfg as cfg
import cv2
import numpy as np
from keras.optimizers import SGD
from densenet121 import DenseNet 
from load_plant_data import load_test_data
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES']='4'
classes = cfg.class_names
num_classes = cfg.num_classes
correct = 0
confusion_mat = np.zeros((num_classes, num_classes)).astype(np.uint64)

# Load data. Please implement your own load_data() module for your own dataset
X_valid, Y_valid, valid_files = load_test_data('train_resize')

# Insert a new dimension for the batch_size
with open('predictions.csv', 'w') as f:
  csv_writer = csv.writer(f, delimiter=',')
  csv_writer.writerow(["File name", "Prediction"])

  # Test pretrained model
  model = DenseNet(reduction=0.5, classes=num_classes, weights_path=cfg.weights_path)
  sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  incorrectfiles = []
  import pdb; pdb.set_trace()
  for i, file_name in enumerate(valid_files):
    im = np.expand_dims(X_valid[i], axis=0)

    # Run prediction
    out = model.predict(im)
    predict = np.argmax(out)

    print(confusion_mat)
    confusion_mat[predict][Y_valid[i]]+=1

    if predict != Y_valid[i]:
      incorrectfiles.append(file_name)
  
    if i%100==0:
      print("Ran for {} images.".format(i))

    csv_writer.writerow([file_name, classes[predict]])

print(incorrectfiles)
print(confusion_mat)
correct = np.sum(confusion_mat.diagonal())
accuracy = (correct * 100)/len(valid_files)
print("Accuracy: {:.2f}".format(accuracy))
