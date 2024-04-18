import numpy as np
import cv2

import os
from typing import Tuple, Union
from multiprocessing.pool import Pool
from pathlib import Path
from glob import iglob
from itertools import chain
import argparse

import albumentations as A

##########################
## PARSER
parser = argparse.ArgumentParser(
    description="Generate augmented images for train or validation according to mocked into the src code"
)

parser.add_argument("--target",
    action='store',
    choices=['train','val'],
    help='Wheater to generate train or val augmented images. It chooses folder automatically. If not provided, --in and --out must be used',
    default=None,
    required=False
)

parser.add_argument('-i','--in',
    action='store',
    help="Input path to a file or directory. If target is provided it will be ignored.",
    default=None,
    required='False',
)

parser.add_argument('-o','--out-img',
    action='store',
    help="Path to a directory to save the images. If target is provided it will be ignored.",
    default=None,
    required=False,
)

parser.add_argument('--out-mask',
    action='store',
    help="Path to a directory to save the masks, can be omitted. If target is provided it will be ignored.",
    default=None,
    required=False,
)

parser.add_argument('-n','--nimages',
    action='store',
    help='Number of images to generate from each original one',
    required=False,
    default=20,
    type=int,
)

parser.add_argument('--limit',
    action='store',
    help='Maximun number of original images to use. This option is intended for test and debug',
    required=False,
    type=int,
)

parser.add_argument('--img-extension',
    action='store',
    choices=['jpg','png'],
    help='Extension to save augmented images (not masks)',
    default='jpg',
    required=False,
)

parser.add_argument('--mask-extension',
    action='store',
    choices=['jpg','png'],
    help='Extension to save augmented masks',
    default='png',
    required=False,
)

parser.add_argument('--n-threads',
    action='store',
    help='Number of pool works to be used',
    default=4,
    required=False,
    type=int,
)

parser.add_argument('--remove',
    action='store_true',
    help='Remove previous images in the augmentation folders'
)

args = vars(parser.parse_args())

##########################
## CONSTANTS
if args['target'] is not None:
    isdir = True
    IMAGES_DIR = os.path.join("dataset","images") # original images
    MASKS_DIR = os.path.join("dataset","masks") # original images

    if args['target']=='train':
        NAME_LIST = os.path.join("dataset","train.txt")
        AUGMENTED_IMAGES_DIR = os.path.join("dataset","augmented_images_train")
        AUGMENTED_MASKS_DIR = os.path.join("dataset","augmented_masks_train")
        
    elif args['target']=='val':
        NAME_LIST = os.path.join("dataset","val.txt")
        AUGMENTED_IMAGES_DIR = os.path.join("dataset","augmented_images_val")
        AUGMENTED_MASKS_DIR = os.path.join("dataset","augmented_masks_val")
    else:
        print("[ERROR] Target must either be 'train' or 'val'")
        exit(-1)
elif args['in'] is not None and args['out-img'] is not None:
    if os.path.isdir(args['in']):
        isdir = True

    elif os.path.isfile(args['in']):
        isdir = False
    else:
        print("[ERROR] --in must be either a file or a path")
        exit(-1)
else:
    print("A target or --in and --out-img must be provided")
    exit(-1)

# execution parameters
NUM_THREADS = args['n_threads']
AUGMENTATION_PER_IMAGE = args["nimages"]
SAVE_EXTENSION = '.'+args['img_extension']
SAVE_MASKS_EXTENSION = '.'+args['mask_extension']

##########################
## TRANSFORMATIONS
transform = A.Compose([
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.1,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT,
        value=None,
        mask_value=None,
        normalized=False, 
        always_apply=False,
        p=0.5
    ),
    A.Rotate(
        limit=360,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        # value=None, # padding value if border_mode = cv2.BORDER_CONSTANT
        # mask_value=None,
        # rotate_method='largest_box', # When dealing with BBOX
        crop_border=False,
        always_apply=True,
        p=0
    ),
    # A.CenterCrop(
    #     p=.1,
    #     width= int(WIDTH*.5),
    #     height= int(WIDTH*.5)
    # ),
    # A.ElasticTransform(
    #     p=.1,
    #     alpha= 40,# magnitude of the displacements
    #     sigma= 40*.05 , # smothness of the displacement
    #     alpha_affine= 40*.03,#WIDTH *.03,
    #     border_mode=cv2.BORDER_REPLICATE,
    #     approximate=True
    # ),
    A.RGBShift(
        r_shift_limit=5,
        g_shift_limit=2,
        b_shift_limit=5,
        always_apply=False,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        p=.5
    ),
    
])

def Augment(img:np.ndarray,mask:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    return transform(image=img,mask=mask).values()

##########################
## APPLY TRANSFORMATIONS

print("\n\n")
def WriteAugment(name:Union[Path,str]) -> None:
    '''
    Write augments for a single image. 

    Function to be sent to a pool of workers
    '''


    name = name.strip()

    img_path = os.path.join(IMAGES_DIR,name)
    masks_path = os.path.join(MASKS_DIR,name)

    img  = cv2.cvtColor(cv2.imread(img_path  ), cv2.COLOR_BGR2RGB  )
    mask = cv2.cvtColor(cv2.imread(masks_path), cv2.COLOR_BGR2GRAY )

    for i in range(AUGMENTATION_PER_IMAGE):
        trans_img , trans_mask = Augment(img.copy(), mask.copy())
        
        trans_img_path  = os.path.join( AUGMENTED_IMAGES_DIR, name[:-4] + f'_{i:02d}' + SAVE_EXTENSION)
        trans_mask_path = os.path.join( AUGMENTED_MASKS_DIR , name[:-4] + f'_{i:02d}' + SAVE_MASKS_EXTENSION)
        
        trans_img  = cv2.cvtColor(trans_img ,cv2.COLOR_RGB2BGR)
        trans_mask = cv2.cvtColor(trans_mask,cv2.COLOR_RGB2BGR)
        cv2.imwrite(trans_img_path , trans_img )
        cv2.imwrite(trans_mask_path, trans_mask)
    
    print(f"{name} ")

# Empty directory
if args['remove']:
    for file in chain(iglob(os.path.join(AUGMENTED_IMAGES_DIR,'*')),iglob(os.path.join(AUGMENTED_MASKS_DIR,'*'))):
        os.remove(file)

if args['limit'] == 0:
    exit(0)

# Perform augmentation
pool = Pool(NUM_THREADS)

with open(NAME_LIST) as file:
    name_list = file.readlines()

    if args['limit'] != None:
        index = np.random.randint(0,len(NAME_LIST),(args['limit'],))
        name_list = np.asarray(name_list)
        name_list = name_list[index]

    pool.map( WriteAugment, name_list )