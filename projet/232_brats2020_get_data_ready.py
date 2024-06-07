# https://youtu.be/oB35sV1npVI
"""
Use this code to get your BRATS 2020 dataset ready for semantic segmentation. 
Code can be divided into a few parts....

#Combine 
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""


import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
import random
import splitfolders

scaler = MinMaxScaler()

TRAIN_DATASET_PATH = 'C:\\Users\\PC\\Desktop\\archive\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\'

# Load sample images and visualize
def load_and_preprocess_image(file_path):
    image = nib.load(file_path).get_fdata()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

def load_and_preprocess_mask(file_path):
    mask = nib.load(file_path).get_fdata()
    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3  # Reassign mask values 4 to 3
    return mask

# Update the file paths correctly
test_image_flair = load_and_preprocess_image(TRAIN_DATASET_PATH + 'BraTS20_Training_355\\BraTS20_Training_355_flair.nii')
test_image_t1 = load_and_preprocess_image(TRAIN_DATASET_PATH + 'BraTS20_Training_355\\BraTS20_Training_355_t1.nii')
test_image_t1ce = load_and_preprocess_image(TRAIN_DATASET_PATH + 'BraTS20_Training_355\\BraTS20_Training_355_t1ce.nii')
test_image_t2 = load_and_preprocess_image(TRAIN_DATASET_PATH + 'BraTS20_Training_355\\BraTS20_Training_355_t2.nii')
test_mask = load_and_preprocess_mask(TRAIN_DATASET_PATH + 'BraTS20_Training_355\\W39_1998.09.19_Segm.nii')

print(np.unique(test_mask))  # Check unique values before reassignment

# Visualize images and mask
n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

# Combine images into channels and divide into patches
combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)
combined_x = combined_x[56:184, 56:184, 13:141]  # Crop to 128x128x128x4
test_mask = test_mask[56:184, 56:184, 13:141]  # Crop mask

# Visualize combined images and mask
n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

# Save combined images and mask
imsave('C:\\Users\\PC\\Desktop\\archive\\BraTS2020_TrainingData\\combined255.tif', combined_x)
np.save('C:\\Users\\PC\\Desktop\\archive\\BraTS2020_TrainingData\\combined255.npy', combined_x)
my_img = np.load('C:\\Users\\PC\\Desktop\\archive\\BraTS2020_TrainingData\\combined255.npy')

test_mask = to_categorical(test_mask, num_classes=4)

# Prepare and save all images and masks
t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*t2.nii'))
t1ce_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*t1ce.nii'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*flair.nii'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*seg.nii'))

for img in range(len(t2_list)):
    print("Now preparing image and masks number:", img)
    
    temp_image_t2 = load_and_preprocess_image(t2_list[img])
    temp_image_t1ce = load_and_preprocess_image(t1ce_list[img])
    temp_image_flair = load_and_preprocess_image(flair_list[img])
    temp_mask = load_and_preprocess_mask(mask_list[img])

    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]  # Crop
    temp_mask = temp_mask[56:184, 56:184, 13:141]  # Crop mask
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(f'C:/Users/PC/Desktop/archive/BraTS2020_TrainingData/input_data_3channels/images/image_{img}.npy', temp_combined_images)
        np.save(f'C:/Users/PC/Desktop/archive/BraTS2020_TrainingData/input_data_3channels/masks/mask_{img}.npy', temp_mask)
    else:
        print("I am useless")

# Split folders into train and validation sets
input_folder = 'C:/Users/PC/Desktop/archive/BraTS2020_TrainingData/input_data_3channels/'
output_folder = 'C:/Users/PC/Desktop/archive/BraTS2020_TrainingData/input_data_128/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)
