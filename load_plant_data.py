'''
    Functions to load the training and validation data.
'''
from config import cfg as cfg
import cv2
import numpy as np
import tqdm
#from keras import backend as K
from keras.utils import np_utils
from PIL import Image
import os

num_classes = cfg.num_classes
UNIT_SCALE = True
SPACE = 50

BASEDIR = '/scratch/cluster-share/kaurd/Contamination_Resnet/ResNet-50-101-152'

def load(set_):
    x, y, files = [], [], []
    path = BASEDIR+'/dataset/'+set_ 
    for name in tqdm.tqdm(os.listdir(path), desc='{:{}}'.format('Load dataset', SPACE), unit_scale=UNIT_SCALE):
        files.append(name)
        import pdb; pdb.set_trace()
        img = Image.open(path+'/'+name) # read as BGR by default
        img = np.asarray(img, np.float64)
        class_ = name.split('_')[0]
        x.append(img)
        y.append(class_)

    x = np.array(x)
    x = mean_subtraction(x)
    y = np.array(y, np.int32)
    return x, y, files

def mean_subtraction(x):    
    mean = [125.96, 127.63, 106.46]
    std = [37.60, 35.39, 37.37]
    for j in range(x.shape[0]):
        for i in range(3):
            x[j][:,:,i] = (x[j][:,:,i]-mean[i])/(std[i]+1e-7)
    return x

def load_data():
    # Load training and validation sets
    (X_train, Y_train, _) = load('train_resize')
    (X_valid, Y_valid, _) = load('val_resize')

    # # Transform targets to keras compatible format
    
    nb_train_samples = Y_train.shape[0]
    nb_valid_samples = Y_valid.shape[0]
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid

def load_test_data(set_):
    # Load training and validation sets
    (X_valid, Y_valid, valid_files) = load(set_)

    #nb_valid_samples = Y_valid.shape[0]
    #Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_valid, Y_valid, valid_files

if __name__ == '__main__':
    load_test_data('train_resize')
