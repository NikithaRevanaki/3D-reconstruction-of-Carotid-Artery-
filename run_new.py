#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, getopt, sys
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.abspath('image_manipulation'))
sys.path.insert(0, os.path.abspath('registration_new'))
sys.path.insert(0, os.path.abspath('volume_manipulation'))

from volume_manipulation.visualisations import visualise_3d_grid
from image_manipulation.image_manip import rename, get_cropped_images
from registration_new.Image_to_Image import register_image_set
from image_manipulation.find_diameter import delete_plate, find_roi
from registration_new.volume_to_volume import volume_registration
from registration_new.slice_to_volume import register_slices
from volume_manipulation.volume_manip import shift_volumes, save_numpy_array, dw_sum, transform_vol, transform_coords, lin_interp

if __name__=='__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],'h',["image_dir="])
        if len(opts)<1 or len(opts)>2:
            print('Wrong number of arguments.')
            print('python3 run.py --iamge_dir=\'<inputfile>\'')
            sys.exit(2)
    except getopt.GetoptError as e:
        print(e.msg)
        print('python3 run.py --iamge_dir=\'<inputfile>\'')
        sys.exit(2)
    image_dir = opts[0][1]
    print('Chosen directory is '+ image_dir+'.')
    path_across = os.path.join("data", image_dir,"transverse")
    path_lengthwise = os.path.join("data", image_dir,"longitudinal")
    rename(path_across) # rename all images in the path for better manipulation
    rename(path_lengthwise)
    transverse_images = get_cropped_images(path_across, switch_direction=False) # crop all images, then put them in a numpy array
    longitudinal_images = get_cropped_images(path_lengthwise, switch_direction=True)
    print('Images were loaded and cropped.')
    Na, Ha, Wa  = transverse_images.shape
    Nl, Hl, Wl = longitudinal_images.shape
    # ----------------image reading and basic preprocessing----------------
    diameter = find_roi(delete_plate(transverse_images.copy(), Na), Na)
    print('Artery diameter was estimated as '+str(diameter)+' pixels.')
    print('Registering images.')
    # ----------------image registration----------------
    lr=1
    step=0.001
    iters=100
    tolerance=0.0005
    transverse_images = register_image_set(transverse_images, Na, lr, step, iters, tolerance) 
    longitudinal_images = register_image_set(longitudinal_images, Nl, lr, step, iters, tolerance)
    print('Done:Images were registered.')
    print('Creating volumes')
    
    # ----------------transverse array creation----------------
    a_spacing = round((Wl-Na)/(Na-1))
    coords_t = np.arange(0, (a_spacing+1)*Na, a_spacing+1).astype(int)
    volume_transverse = lin_interp(transverse_images, coords_t)

    # ----------------longitudinal array creation----------------
    l_spacing = round((diameter-Nl)/(Nl-1))
    coords_l = np.arange(0, (l_spacing+1)*Nl, l_spacing+1).astype(int)
    volume_long = lin_interp(longitudinal_images, coords_l)
    volume_long = np.transpose(volume_long, (2, 1, 0)) # this is done to match the axes of both arrays

    print('Done: two volumes were created')
    # ----------------volume registration----------------
    print('Starting volume registration.')
    volume_transverse_reg, across_volume_transform, long_transform = volume_registration(volume_long, volume_transverse)
    volume_long_reg = transform_vol(long_transform, volume_transverse, volume_long).astype(np.uint8)
    print('Done: volume registration finished.')
    for i in range(2):
        # ----------------registration of slices in volumes----------------
        print('Starting slice-volume registration.')
        coords_t = register_slices(volume_transverse, volume_long_reg, Na, coords_t, a_spacing)
        transverse_transposed = np.transpose(volume_transverse_reg, (2, 1, 0))
        coords_l = register_slices(np.transpose(volume_long, (2, 1, 0)), transverse_transposed, Nl, coords_l, l_spacing)
        # ----------------creating volumes----------------
        print('Creating volumes.')
        z_shift_t, coords_t = transform_coords(coords_t)
        volume_transverse = lin_interp(transverse_images, coords_t)
        z_shift_l, coords_l = transform_coords(coords_l)
        volume_long = lin_interp(longitudinal_images, coords_l)
        volume_transverse =  np.transpose(volume_transverse, (2, 1, 0))
        volume_transverse, volume_long = shift_volumes(z_shift_l, volume_transverse, volume_long)
        volume_transverse =  np.transpose(volume_transverse, (2, 1, 0))

        volume_long = np.transpose(volume_long, (2, 1, 0))
        volume_long, volume_transverse = shift_volumes(z_shift_t, volume_long, volume_transverse)

        #   ----------------volume registration----------------
        orig_transform = across_volume_transform
        volume_transverse_reg = transform_vol(across_volume_transform, volume_transverse, volume_transverse).astype(np.uint8)
        volume_transverse_reg, new_across_volume_transform, new_long_transform = volume_registration(volume_long, volume_transverse_reg)
        across_volume_transform = sitk.CompositeTransform([new_across_volume_transform, across_volume_transform])
        long_transform = across_volume_transform.GetInverse()
        volume_long_reg = transform_vol(long_transform, volume_transverse, volume_long).astype(np.uint8)
    
    print('Done: slices were registered.')
    print('Creating a final volume.')
    r = dw_sum(volume_long, volume_transverse, coords_l, coords_t, long_transform)
    save_numpy_array(r, image_dir)
    visualise_3d_grid(r)


# In[ ]:




