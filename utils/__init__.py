import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

import os
from typing import List, Tuple, Union

'''
Some utilities for iamge manipulation

'''
__name__ = "Utils"
__all__ = [
    'display',
    'draw_grid',
    'augmented_display',
    'multipredict',
    'read_image',
]

def read_image(image_path:str,shape:Union[List[int],Tuple[int,int,int]]) -> tf.Tensor:
    '''
    Read image from its path. Returns an uint8 tensor.

    Params
    -------
        image_path: str
    Path to the image to open

        shape
    Tuple or list with (width, height, channels). If 
    the image shape doesn't have the specified width
    and height it will be resized.

    Returns
    --------
        tensor (width, height, channels) with the image in RGB format
    '''
    if len(shape)!=3:
        raise ValueError("The shape must be a tuple/list (width,height,channels)") 

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=shape[2])
    image.set_shape([None, None, shape[2]])
    image = tf.image.resize(images=image, size=shape[0:2])
    image = tf.cast(image, dtype=tf.uint8)

    return image

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


def color_map(mask:tf.Tensor) -> np.ndarray:
    '''
    Turn an 1-encoded gray image into a colored image
    '''
    background_color = (0,0,0) # black
    clear_sky = (92, 179, 255) # clear blue
    thin_cloud = (255, 243, 245) # shiny gray
    tick_cloud = (113, 125, 150) # deep gray
    sun = (255, 242, 0) # yellow
    
    colored_image = np.empty((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    
    
    for class_index, color in enumerate([background_color,clear_sky,sun,tick_cloud,thin_cloud]):
        indices = np.where(mask == class_index)
        colored_image[indices[0],indices[1],:] = color
    return colored_image

def multipredict(img_list:List[str],mask_list:List[str],model:tf.keras.Model) -> None:
    if len(img_list) != len(mask_list):
        raise ValueError('There must be the same number os masks and images')
    if len(img_list) == 0:
        raise ValueError('There must be at least one image')
    
    fig, ax = plt.subplots(len(img_list),3,figsize=(12,4*len(img_list)+3))
    index = 0 # row counter
    for img_path, mask_path in zip(img_list,mask_list):
        try: 
            # "When loading the model with tf.keras.models.load_model( )
            # the model.input is a list with a single element. A list
            # doesn't have shape attribute. When defining the model from
            # scratch, model.input is the input layer
            shape = model.input.shape[1:]
        except AttributeError:
            shape = model.input[0].shape[1:]

        img = read_image(img_path,shape)
        ax[index,0].imshow( img )
        ax[index,0].set_xticks([])
        ax[index,0].set_yticks([])
        
        pred = model.predict( tf.expand_dims(img,0) )
        ax[index,1].imshow( color_map(tf.argmax(pred[0,:,:,:],2)) )
        ax[index,1].set_xticks([])
        ax[index,1].set_yticks([])
        
        ax[index,2].imshow(color_map(read_image(mask_path,shape)[:,:,0]))
        ax[index,2].set_xticks([])
        ax[index,2].set_yticks([])
        
        index += 1
    
    ax[0,0].set_title("Image")
    ax[0,1].set_title("Inference")
    ax[0,2].set_title("Ground truth")