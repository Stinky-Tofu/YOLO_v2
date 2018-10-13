# coding:utf-8

# data
DATA_DIR = '../data' # Data and project need to be in the same directory
MODEL_DIR = './model'
MODEL_FILE = 'yolo_coco_initial.ckpt'
LOG_DIR = './log'

# network
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
ANCHOR = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

# compute environment
GPU = ''

IMAGE_SIZE = 416    # The size of the input images

LEARN_RATE_BASE = 0.0001   # The learn_rate of training
MOVING_AVE_DECAY = 0.9995 # the moving average decay of weights
MAX_PERIODS = 100    # The max train periods
SAVE_ITER = 2

BOX_PRE_CELL = 5    # The number of BoundingBoxs predicted by each grid
CELL_SIZE = 13      # The size of the last layer  #(batch_size, 13, 13, ?)
BATCH_SIZE = 16     # The batch size of each training

PROB_THRESHOLD = 0.3    # The threshold of the probability of the classes
NMS_THRESHOLD = 0.5     # The threshold of the IOU when implement NMS

