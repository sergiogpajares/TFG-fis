import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
from typing import List, Tuple, Union

'''
Some utilities for iamge manipulation

'''
__name__ = "Utils"
__all__ = [
    'display',
    'draw_grid',
    'augmented_display'
]

def display(images:Union[List[Union[np.ndarray,str]],str,np.ndarray],ncols:int=3) -> None:
    '''
    Display an image or a list or images
    '''

    if isinstance(images,List):
        nimages = len(images)
        nrows=nimages//ncols +1
        fig, ax = plt.subplots(nrows, ncols, figsize=(3.3*ncols,3.3*nrows))

        if nrows == 1:
            for j in range(ncols):
                    ax[j].axis("off")

                    if j >= nimages:
                        continue
                    
                    if isinstance(images[j],str):
                        img = cv2.imread( images[j] )
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    else:
                        img = images[j] 
                    ax[j].imshow(img)
        else:
            for i in range(nrows):
                for j in range(ncols):
                    ax[i,j].axis("off")
                    
                    n_img = i*ncols + j
                    if n_img >= nimages:
                        continue
                    
                    if isinstance(images[n_img],str):
                        img = cv2.imread( images[n_img] )
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    else:
                        img = images[n_img] 
                    ax[i,j].imshow(img)
                    
    else:
        nrows = 1
        fig = plt.figure(figsize=(5,5))
        
        if isinstance(images,str):
            img = cv2.imread( images )
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img = images

        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()

def augmented_display(image:np.ndarray, mask:np.ndarray, original_image:Union[np.ndarray,None]=None, original_mask:Union[np.ndarray,None]=None) -> None:
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[0].axis('off')

        ax[1].imshow(mask)
        ax[0].axis('off')
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[0,0].axis('off')

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[1,0].axis('off')

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[0,1].axis('off')

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        ax[1,1].axis('off')
    
    plt.tight_layout()

def draw_grid(im:np.ndarray, grid_size:int) -> np.ndarray:
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))
    return im