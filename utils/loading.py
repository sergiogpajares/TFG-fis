import tensorflow as tf

from . import read_image

from typing import Union, Tuple, List

'''
This submodule contains some requiered functions to generate a TensorFlow dataset.

'''

__all__ = [
    'read_image',
    'read_mask',
    'data_generator',
    'map_weights'
]

def read_image(image_path:str,shape:Union[List[int],Tuple[int,int,int],None]=None) -> tf.Tensor:
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
    if shape is not None:
        if len(shape)!=3:
            raise ValueError("The shape must be a tuple/list (width,height,channels)") 

    image = tf.io.read_file(image_path)
    if shape is not None:
        image = tf.image.decode_png(image, channels=shape[2])
        image.set_shape([None, None, shape[2]])
        image = tf.image.resize(images=image, size=shape[0:2])
        image = tf.cast(image, dtype=tf.uint8)
    else:
        image = tf.image.decode_image(image,dtype=tf.uint8)


    return image

def read_mask(mask_path:str, num_classes:int, img_shape:Union[List[int],Tuple[int,int,int]]) -> tf.Tensor:
    '''
    Read mask from its path. Returns a one hot-encoded tensor
    '''
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(images=mask, size=img_shape[:2])
    mask = tf.cast(mask, dtype=tf.uint8)

    # Other classified as number 8 (check)
    if num_classes > 1:
        mask = tf.squeeze(mask,axis=2) #remove extra axis
        mask = tf.one_hot(mask, depth = num_classes)
        
    return mask

def _load_data(image:str, mask:str,num_classes:int,img_shape:Union[List[int],Tuple[int,int,int]]) -> Tuple[tf.Tensor,tf.Tensor]:
    '''
    Auxiliar function to read both image and mask
    '''
    
    image = read_image(image,img_shape)
    mask = read_mask(mask,num_classes,img_shape)
    
    return image, mask

def map_weights(image:tf.Tensor, label:tf.Tensor, class_weights:tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    '''
    Assuming label is one-hot encoded, calculate weights based on the class
    '''
    
    weights = tf.reduce_sum(label * class_weights, axis=-1)  # Calculate weights based on class
    
    return image, label, weights

def data_generator(
    image_list:List[str],
    mask_list:List[str],
    batch_size:int,
    num_classes:int,
    img_shape:Union[List[int],Tuple[int,int,int]],
    ) -> tf.data.Dataset:
    '''
    Return a dataset from a list of images paths
    '''
    
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(lambda image,mask: _load_data(image,mask,num_classes,img_shape), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset
