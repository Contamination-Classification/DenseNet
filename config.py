from easydict import EasyDict
cfg = EasyDict()

# number of classes
cfg.num_classes = 2 

# Use pre-trained weights for Tensorflow backend
cfg.weights_path = 'model.h5'

# define classes
cfg.class_names = ['C', 'NC']

cfg.RGB_DIR = 'all_shuffled_030920_a3/'

# configuration for 3X4 grid.
cfg.CROP_DIMS = (310, 515, 1750, 1610) #tw, th, bw, bh

cfg.GRIDW = 4

cfg.GRIDH = 3

cfg.OUTPUT = "contamination_results.csv"

# k nearest neighbor classes
classes = {(128,128,128):0, (0, 255, 0):1, (150,75,0):1, (181,101,29):1, (187,144,103):1, (255,182,193):2, (255,255,0):2, (255,255,51):2, (255,140,0):2, (0,0,0):2, (255,255,255):2}
