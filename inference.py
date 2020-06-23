'''
    Author - Damanpreet Kaur
    Date - 06/09/2020
    Purpose - Split explants in the image and classify them as contaminated/not contaminated.
        input - rgb image directory.
        output - csv file with output labels 
                 (C - Contamination, NC - Non-Contaminated, M - Missing)
'''
import argparse
import numpy as np
from utils import RGBPreprocess, missing_expant
from config import cfg as cfg
import numpy as np
from keras.optimizers import SGD
from densenet121 import DenseNet 
from load_plant_data import load_test_data
import os
import cv2
import csv
from os import path as osp
from PIL import Image
from load_plant_data import mean_subtraction

os.environ['CUDA_VISIBLE_DEVICES']='0'
INVALID_GRID_TYPE = 'Grid type is Invalid.'

def get_arguments():
    """
        Parse all the command line arguments.
    """
    parser = argparse.ArgumentParser(description="contamination-components")
    parser.add_argument("--img-list", required=True, help="Input list that contains complete path to the images")
    parser.add_argument("--output_file", default=cfg.OUTPUT, help="Output file name")
    parser.add_argument("--grid_type", default=12, help="Type of grid (grid type 1 - 12 explants)")
    return parser.parse_args()

def main():
    # create directory to save the preprocessed images
    if not os.path.exists('preprocessed_images'):
        os.mkdir('preprocessed_images')
        
    print("Reading Arguments: ")
    args = get_arguments()

    try:
        images = open(args.img_list).readlines()
    except Exception as e:
        print("Unable to read the image list")
        print(e)
        exit()

    #images = []
    #for r, d, f in os.walk(args.img_dir):
    #    for file in f:
    #        if '.jpg' in file:
    #            images.append(osp.join(r, file))

    print("Processing {} images.".format(len(images)))

    header = ['image_name']
    header.extend(list(range(1,13)))
    rows = []

    # Initialize the pretrained model
    print("Intializing the pretrained model")
    model = DenseNet(reduction=0.5, classes=cfg.num_classes, weights_path=cfg.weights_path)
    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded sucessfully!")

    for img_name in images:
        img_name = img_name.strip()
        image_row = dict()
        image_row['image_name'] = img_name.split('/')[-1]
        image = cv2.imread(img_name)
        (h, w) = image.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        im_name = osp.join('preprocessed_images', img_name.split('/')[-1])
        cv2.imwrite(im_name, image)

        if args.grid_type == 12:
            crop_dims, gridw, gridh = cfg.CROP_DIMS, cfg.GRIDW, cfg.GRIDH
        else:
            raise ValueError(INVALID_GRID_TYPE)

        # Pre-process the image.
        rgb = RGBPreprocess(crop_dims)
        data = rgb.process_img(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gridh, gridw)

        # run for each grid location.
        for i, im in enumerate(data):
            # preprocess the image
            im = Image.fromarray(im)
            im = im.resize((224, 224), Image.LANCZOS)
            im = np.asarray(im, np.float64)
            im = np.expand_dims(im, axis=0)
            im = mean_subtraction(im)
                
            # Run prediction
            out = model.predict(im)
            predict = np.argmax(out)

            # write prediction to the row dictionary.
            image_row[i+1] = cfg.class_names[predict]
        rows.append(image_row)

    # write to csv
    with open(args.output_file, 'w') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    main()
