import numpy as np
import cv2

import os
from typing import List, Tuple, Union, Literal
from pathlib import Path
from glob import iglob, glob
import argparse

from utils import color_map, read_image

##########################
## PARSER
parser = argparse.ArgumentParser(
    description="Generate augmented images for train or validation according to mocked into the src code"
)

parser.add_argument('-i','--in',
    action='store',
    type = str,
    help='Input masks path',
    required=True
)

parser.add_argument('-o','--out',
    action='store',
    help='Output mask path',
    type=str,
    required=True
)

parser.add_argument('--ext','--extension',
    action='store',
    choices=['jpg','png'],
    help='Extension to save augmented masks',
    default='png',
    required=False,
)

parser.add_argument('--shape',
    action='store',
    help='width,height to resize the image to',
    nargs=2,
    type=int,
    required=False,
    default = None,
)

parser.add_argument('--limit',
    action='store',
    type=int,
    default=None,
    required=False
)

parser.add_argument('--remove',
    action='store_true',
    help='Remove previous images in the augmentation folders'
)

args = vars(parser.parse_args())

###########################
## LET'S ROCK

def write_image(
        path:Union[Path,str],
        output_base_path:Union[Path,str],
        extension:Literal['png','jpg']='png',
        shape:Union[List[int],Tuple[int,int],None]=None,
    ) -> None:
    '''
    Write a colored mask into the output path from a path
    to a raw mask.

    Params
    --------
        path: str|Path
    Path to the mask that is to be colored

        output_base_path: str|Path
    Directory to save the masks

        extension: 'png' or 'jpg'
    Extension to save the colored mask

        shape: array-like (width, height)
    Resize the image to that shape
    '''
    # Read image
    if shape is not None:
        mask = read_image(path,(shape[0],shape[1],3),'cv2')
    else:
        mask = read_image(path,shape,'cv2')

    # Color
    mask = color_map(mask)

    # Write
    output_path = os.path.join(output_base_path, Path(path).stem +'.'+extension)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(output_path,mask):
        raise RuntimeError('Image could not be saved')

    
## Cheking input

if not os.path.isdir( args['out'] ):
    print("[ERROR] output path must be a directory")
    exit(-1)

if not os.path.isdir( args['in'] ):
    print("[ERROR] input path must be a directory")
    exit(-1)

## Removing unused
if args['remove']:
    for file in iglob(os.path.join(args['out'],'*'+args['ext'])):
        os.remove(file)

if args['limit'] == 0:
    exit(0)

## Write images
name_list = glob(os.path.join(args['in'],'*') )

if args['limit'] != None:
    index = np.random.randint(0,len(name_list),(args['limit'],))
    name_list = np.asarray(name_list)
    name_list = name_list[index]

for path in name_list:
    write_image(path,args['out'],args['ext'],args['shape'])