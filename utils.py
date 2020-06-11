"""
    Utility functions for finding the missing explants.
"""
import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import config as cfg

class RGBPreprocess:
    def __init__(self, crop_dims):
        self.tw, self.th, self.bw, self.bh = crop_dims

    def process_img(self, image, gridh, gridw):
        '''
            Return the list of cropped explants.
        '''
        print("Preprocess the image.")

        # Resize the image and split the image.
        self.h, self.w = 2000, 2000
        image = cv2.resize(image, (self.h, self.w))
        img = image[self.th:self.bh, self.tw:self.bw]
        height, width, _ = img.shape

        # grid dimensions
        GRID_RANGE_W = math.ceil(width / gridw)
        GRID_RANGE_H = math.ceil(height / gridh)

        count, data = 0, []
        for h in range(0, height-1, GRID_RANGE_H):
            for w in range(0, width-1, GRID_RANGE_W):
                count+=1
                image = img[h:h+GRID_RANGE_H, w:w+GRID_RANGE_W]
                data.append(image)

        return data

# Function to find the Eucledian distance
def eucledian_dis(pt1, pt2):
    return np.linalg.norm(pt1-pt2)


def knearest_neighbor(image):
    '''
        Function to find the K-nearest neighbor.
        Steps:
        1. classify each pixel into categories.
        2. If the image is contaminated, return 2
            if the explant is missing, return 0
            else, return 1
    '''
    image = cv2.medianBlur(image, 5) # apply the median filter to reduce the noise.
    ht, wd, _ = image.shape
    mask = np.zeros([ht, wd])

    # check for every pixel in the image
    for r1 in range(ht):
        for c1 in range(wd):
            boundary = False
            best = float('inf')
            chosencls = 0
            for ref in cfg.classes:
                dist = eucledian_dis(image[r1][c1], np.array(ref))  # check the class to which the pixel belongs
                if dist < best:
                    if ref == (255,255,255):                        # check for boundary pixels
                        for pt in np.array([[0, 0], [0, wd], [ht, 0], [ht, wd]]):
                            if eucledian_dis(np.array([r1, c1]), pt) < 100:
                                boundary = True
                    if not boundary:
                        best=dist
                        chosencls=cfg.classes[ref]

            mask[r1, c1]=chosencls

    return mask.astype(int)


def missing_expant(image):
    mask = knearest_neighbor(image)             # apply k-nearest neighbor and obtain a mask
    counts = np.bincount(mask.flatten(), minlength=3)
    if counts[2] > 2500 or counts[1] > 2500:
        return False
    return True    