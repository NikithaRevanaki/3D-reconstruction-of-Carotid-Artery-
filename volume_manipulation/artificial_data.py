#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def create_data(ts, ls):
    """creates the artificial volume and evenly spaced slices
       for evaluation

    Returns:
        ndarray: created data and slices
    """
    volume = np.empty([350, 350, 600], dtype=np.uint8)
    x, y = np.mgrid[0:350, 0:350]
    circle = np.sqrt((x - 150) ** 2 + (y - 150) ** 2)
    img_array  = np.logical_and(circle < 125, circle > 80)*100
    img_array  -= np.logical_and(circle < 125-10, circle > 80+10)*30
    volume[:, :, 0]= img_array
    for i in range(599):
        img_array = np.logical_and(circle < 125-i/12, circle > 80-i/12)*100
        img_array -= np.logical_and(circle < 125-i/12-10, circle > 80-i/12+10)*30
        volume[:, :, i]= img_array
    volume[300:340,:] = 200
    volume = np.transpose(volume, (2, 0, 1))
    transverse = volume.copy()[::ts, :, :]
    longitudinal = volume.copy()[:, :, ::ls]
    longitudinal = np.transpose(longitudinal, (2, 1, 0))
    return volume, transverse[1:-1, :, :], longitudinal[3:-5, :, :]

