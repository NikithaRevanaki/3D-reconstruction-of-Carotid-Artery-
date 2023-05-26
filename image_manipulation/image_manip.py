#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os
import numpy as np

TOP_CROP = 100
LEFT_CROP = 185
BOTTOM_CROP = 10
RIGHT_CROP = 35

def rename(image_path):
    """renames all images in the path, so that they have only 
       numeric symbols in the name and can be sorted by their name
       (for example changes 1.png to 01.png)

    Args:
        image_path (str): path of the images
    """
    if not os.path.exists(image_path):
        print('Chosen path does not exits.')
        exit(2)
    images_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    for img_name in images_names:
        new_name = '{:0>2}'.format(img_name.split('.')[0].replace('x', ''))
        os.rename(os.path.join(image_path,img_name), os.path.join(image_path, new_name+'.'+img_name.split('.')[1]))

def get_cropped_images(image_path, switch_direction):
    """crops all images in a given directory by a constant

    Args:
        image_path (string): path of the images to be cropped

    Returns:
        ndarray: 3D array of the cropped images
    """
    if not os.path.exists(image_path):
        print("Path does not exist, cannot create a 3D volume.")
        exit()
    else:
        image_names = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        image_names.sort(reverse=switch_direction) # longitudinal images need to be read in the opposite direction
        array_3d = np.array([np.array(Image.open(os.path.join(image_path, image)).convert('L')) for image in image_names])
        array_3d = array_3d[:, TOP_CROP:-BOTTOM_CROP, LEFT_CROP:-RIGHT_CROP]
    return array_3d.astype(np.uint8)

def series_to_3d_array(image_path):
    """turns given images into a numpy array 

    Args:
        image_path (str): name of the path of input images

    Returns:
        ndarray: 3d array of a set of images
    """
    if not os.path.exists(image_path):
        print("Path does not exist")
        exit()
    else:
        images = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        images.sort()
        array_3d = np.array([np.array(Image.open(os.path.join(image_path, image)).convert('L')) for image in images])
    return array_3d.astype(np.uint8)


# In[ ]:




