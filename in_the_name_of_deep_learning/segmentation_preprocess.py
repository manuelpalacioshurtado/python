import numpy as np
import os
from skimage import io
from skimage.transform import resize
import cv2
import pickle
import matplotlib.pyplot as plt

# parameters that you should set before running this script
voc_root_folder = "/Users/ericmassip/Projects/MAI/2nd_semester/CV/Project/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 128    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)


def build_segmentation_dataset(imagesets_filename):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    imagesets_file = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/" + imagesets_filename)
    with open(imagesets_file) as file:
        train_files = file.read().splitlines()

    x_image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    x_image_filenames = [os.path.join(x_image_folder, file) for f in train_files for file in os.listdir(x_image_folder) if f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in x_image_filenames]).astype('float32')

    y_image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationObject/")
    y_image_filenames = [os.path.join(y_image_folder, file) for f in train_files for file in os.listdir(y_image_folder) if f in file]
    y = np.array([resize(io.imread(img_f, as_gray=True), (image_size, image_size, 3)) for img_f in y_image_filenames]).astype('float32')

    return x, y


x_train, y_train = build_segmentation_dataset('train.txt')
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_segmentation_dataset('val.txt')
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))


def convert_to_grayscale(images):
    return np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])


x_train_gray = convert_to_grayscale(x_train)
y_train_gray = np.where(convert_to_grayscale(y_train) > 0, 1, 0)
x_val_gray = convert_to_grayscale(x_val)
y_val_gray = np.where(convert_to_grayscale(y_val) > 0, 1, 0)

pickle.dump(x_train_gray, open('x_train_seg_128.P', 'wb'))
pickle.dump(y_train_gray, open('y_train_seg_128.P', 'wb'))
pickle.dump(x_val_gray, open('x_val_seg_128.P', 'wb'))
pickle.dump(y_val_gray, open('y_val_seg_128.P', 'wb'))