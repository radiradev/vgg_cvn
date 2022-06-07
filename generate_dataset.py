from socket import if_indextoname
import numpy as np
import glob
import ast
import ntpath
import pickle
import configparser
import logging
import sys
import time
import random
import zlib

from scipy.interpolate import interp1d
from sklearn.utils import class_weight
from collections import Counter
import pandas as pd
"""
****************************************
************** PARAMETERS **************
****************************************
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

config = configparser.ConfigParser()
config.read("config.ini")

# random

SEED = int(config["random"]["seed"])

if SEED == -1:
    SEED = int(time.time())

np.random.seed(SEED)

# images

IMAGES_PATH = config["images"]["path"]
VIEWS = int(config["images"]["views"])
PLANES = int(config["images"]["planes"])
CELLS = int(config["images"]["cells"])

# dataset

DATASET_PATH = config["dataset"]["path"]
PARTITION_PREFIX = config["dataset"]["partition_prefix"]
LABELS_PREFIX = config["dataset"]["labels_prefix"]
UNIFORM = ast.literal_eval(config["dataset"]["uniform"])

# model

OUTPUTS = int(config["model"]["outputs"])

# train

TRAIN_FRACTION = float(config["train"]["fraction"])
WEIGHTED_LOSS_FUNCTION = ast.literal_eval(config["train"]["weighted_loss_function"])
CLASS_WEIGHTS_PREFIX = config["train"]["class_weights_prefix"]

# validation

VALIDATION_FRACTION = float(config["validation"]["fraction"])

# test

TEST_FRACTION = float(config["test"]["fraction"])

if (TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION) > 1:
    logging.error("(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION) must be <= 1")
    exit(-1)

# Energy mapping
def energyMapping(fNuEnergy):
    if fNuEnergy > 8:
        return 1.0
    m = interp1d([0, 8], [0, 1])
    return m(fNuEnergy)


# Return 3 if value > 2
def normalize(value):
    if value > 2:
        return 3
    return value


# Return 1 if N < 0 else 0
def normalize2(value):
    if value > 50000:
        return 1
    # if value < 0:
    #    return 1
    return 0


count_flavour = [0] * 4
count_category = [0] * 14

"""
****************************************
*************** DATASETS ***************
****************************************
"""

partition = {
    "train": [],
    "validation": [],
    "test": [],
}  # Train, validation, and test IDs
labels = {}  # ID : label
y_train = []
y1_class_weights = []
y2_class_weights = []

if UNIFORM:
    pass

# Iterate through label folders

logging.info("Filling datasets...")

count_neutrinos = 0
count_antineutrinos = 0
count_empty_views = 0
count_empty_events = 0
count_less_10nonzero_views = 0
count_less_10nonzero_events = 0

labels_full_path = '/afs/cern.ch/work/r/rradev/public/vgg_cvn/labels.p'

with open(labels_full_path, "rb") as l_file:
    labels_full = pickle.load(l_file)

for images_path in glob.iglob(IMAGES_PATH + "/*"):

    count_train, count_val, count_test = (0, 0, 0)

    print(images_path)

    files = list(glob.iglob(images_path + "/images/*"))
    random.shuffle(files)

    for imagefile in files:
        # print imagefile
        ID = imagefile.split("/")[-1][:-3]

        # print infofile


        random_value = np.random.uniform(0, 1)

        with open(imagefile, "rb") as image_file:
            pixels = np.fromstring(
                zlib.decompress(image_file.read()), dtype=np.uint8, sep=""
            ).reshape(VIEWS, PLANES, CELLS)

            # pixels = np.load(self.images_path + '/' + labels[ID] + '/' + ID + '.npy')

            views = [None] * VIEWS
            empty_view = [0, 0, 0]
            non_empty_view = [0, 0, 0]

            # Events that contain any view with less than 10 non-zero pixels are filtered out.
            # This is to remove empty and almost empty images from the set
            count_empty = 0
            count_less_10nonzero = 0
            for i in range(len(views)):
                views[i] = pixels[i, :, :].reshape(PLANES, CELLS, 1)
                maxi = np.max(views[i])  # max pixel value (normally 255)
                mini = np.min(views[i])  # min pixel value (normally 0)
                nonzero = np.count_nonzero(views[i])  # non-zero pixel values
                total = np.sum(views[i])
                avg = np.mean(views[i])
                if nonzero == 0:
                    count_empty += 1
                    count_empty_views += 1
                if nonzero < 10:
                    count_less_10nonzero += 1
                    count_less_10nonzero_views += 1
            if count_empty == len(views):
                count_empty_events += 1
            if count_less_10nonzero > 0:
                count_less_10nonzero_events += 1
                continue

        # Fill training set
        if random_value < TRAIN_FRACTION:
            partition["train"].append(ID)
            count_train += 1
        # Fill validation set
        elif random_value < (TRAIN_FRACTION + VALIDATION_FRACTION):
            partition["validation"].append(ID)
            count_val += 1
        # Fill test set
        elif random_value < (TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION):
            partition["test"].append(ID)
            count_test += 1

        # Label
        print(ID)
        labels[ID] = labels_full[ID]
        print(count_train)

    logging.debug("%d train images", count_train)
    logging.debug("%d val images", count_val)
    logging.debug("%d test images", count_test)
    logging.debug("%d total images", count_train + count_val + count_test)

# Print dataset statistics

print(count_flavour)
print(count_category)

logging.info("Number of neutrino events: %d", count_neutrinos)
logging.info("Number of antineutrino events: %d", count_antineutrinos)

logging.info("Number of empty views: %d", count_empty_views)
logging.info("Number of views with <10 non-zero pixels: %d", count_less_10nonzero_views)
logging.info("Number of empty events: %d", count_empty_events)
logging.info(
    "Number of events with at least one view with <10 non-zero pixels: %d",
    count_less_10nonzero_events,
)

logging.info("Number of training examples: %d", len(partition["train"]))
logging.info("Number of validation examples: %d", len(partition["validation"]))
logging.info("Number of test examples: %d", len(partition["test"]))

# Serialize partition and labels

logging.info("Serializing datasets...")

with open(DATASET_PATH + PARTITION_PREFIX + ".p", "wb") as partition_file:
    pickle.dump(partition, partition_file)

with open(DATASET_PATH + LABELS_PREFIX + ".p", "wb") as labels_file:
    pickle.dump(labels, labels_file)
