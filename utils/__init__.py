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
    'display'
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