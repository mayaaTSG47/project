import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import random

def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:
        if image_name.split('.')[-1] == 'nii':
            image_path = os.path.join(img_dir, image_name)
            image = nib.load(image_path).get_fdata()
            images.append(image)
    images = np.array(images)
    return images

def normalize_img(image):
    image = image - np.min(image)
    if np.max(image) > 0:
        image = image / np.max(image)
    return image

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true  
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            # Normalize images
            X = np.array([normalize_img(img) for img in X])
            
            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

############################################

# Test the generator
train_img_dir = "C:\\Users\\PC\\Desktop\\archive\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_022\\"
train_mask_dir = "C:\\Users\\PC\\Desktop\\archive\\BraTS2020_ValidationData\\MICCAI_BraTS2020_ValidationData\\BraTS20_Validation_022\\"
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]

# Fixing the indexing based on the dimensions of the loaded images and masks
if test_img.ndim == 3:  # Check if the image is 3D (H, W, C)
    n_slice = random.randint(0, test_img.shape[2] - 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(test_img[:, :, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(test_img[:, :, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:, :, n_slice])
    plt.title('Mask')
    plt.show()
elif test_img.ndim == 4:  # Check if the image is 4D (H, W, S, C)
    n_slice = random.randint(0, test_img.shape[2] - 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:, :, n_slice])
    plt.title('Mask')
    plt.show()
else:
    print("Unexpected image dimensions:", test_img.shape)
