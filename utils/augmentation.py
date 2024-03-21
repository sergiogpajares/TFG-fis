import cv2
import numpy as np

from typing import Tuple, Union

'''
General utils for image augmentation

Author: Sergio García Pajares (sergiogaciapajares@gmail.com)

Some of this code base is partially based on https://towardsdatascience.com/image-augmentation-mastering-15-techniques-and-useful-functions-with-python-codes-44c3f8c1ea1f
'''
__name__="Augmentation utils"
__all__=["Perspective","Contrast","UniformNoise","GaussianNoise","Vignetting","LensDistortion"]

class Perspective(object):
    "Random Perspective transform (Translation, Rotation, Scalation, Shering)"

    def __init__(self,
            max_ratio_translation:Tuple[float,float,float]=(0.2, 0.2, 0),
            max_rotation:Tuple[float,float,float]=(10, 10, 360),
            max_scale:Tuple[float,float,float]=(0.1, 0.1, 0.2),
            max_shearing:Tuple[float,float,float]=(15, 15, 5)
        ):

        self.max_ratio_translation = np.array(max_ratio_translation)
        self.max_rotation = np.array(max_rotation)
        self.max_scale = np.array(max_scale)
        self.max_shearing = np.array(max_shearing)

    def __call__(self, X:np.ndarray, Y:Union[np.ndarray,None] = None)-> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply a random Perspective transform considering:
            1. Translation
            2. Rotation
            3. Scalation
            4. Shering
        
        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image
        '''

        # get the height and the width of the image
        h, w = X.shape[:2]
        max_translation = self.max_ratio_translation * np.array([w, h, 1])
        # get the values on each axis
        t_x, t_y, t_z = np.random.uniform(-1, 1, 3) * max_translation
        r_x, r_y, r_z = np.random.uniform(-1, 1, 3) * self.max_rotation
        sc_x, sc_y, sc_z = np.random.uniform(-1, 1, 3) * self.max_scale + 1
        sh_x, sh_y, sh_z = np.random.uniform(-1, 1, 3) * self.max_shearing

        # convert degree angles to rad
        theta_rx = np.deg2rad(r_x)
        theta_ry = np.deg2rad(r_y)
        theta_rz = np.deg2rad(r_z)
        theta_shx = np.deg2rad(sh_x)
        theta_shy = np.deg2rad(sh_y)
        theta_shz = np.deg2rad(sh_z)


        # compute its diagonal
        diag = (h ** 2 + w ** 2) ** 0.5
        # compute the focal length
        f = diag
        if np.sin(theta_rz) != 0:
            f /= 2 * np.sin(theta_rz)

        # set the image from cartesian to projective dimension
        H_M = np.array([[1, 0, -w / 2],
                        [0, 1, -h / 2],
                        [0, 0,      1],
                        [0, 0,      1]])
        # set the image projective to carrtesian dimension
        Hp_M = np.array([[f, 0, w / 2, 0],
                         [0, f, h / 2, 0],
                         [0, 0,     1, 0]])

        # adjust the translation on z
        t_z = (f - t_z) / sc_z ** 2
        # translation matrix to translate the image
        T_M = np.array([[1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0,  1]])

        # calculate cos and sin of angles
        sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
        sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
        sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
        # get the rotation matrix on x axis
        R_Mx = np.array([[1,      0,       0, 0],
                         [0, cos_rx, -sin_rx, 0],
                         [0, sin_rx,  cos_rx, 0],
                         [0,      0,       0, 1]])
        # get the rotation matrix on y axis
        R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                         [     0, 1,       0, 0],
                         [sin_ry, 0,  cos_ry, 0],
                         [     0, 0,       0, 1]])
        # get the rotation matrix on z axis
        R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                         [sin_rz,  cos_rz, 0, 0],
                         [     0,       0, 1, 0],
                         [     0,       0, 0, 1]])
        # compute the full rotation matrix
        R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

        # get the scaling matrix
        Sc_M = np.array([[sc_x,     0,    0, 0],
                         [   0,  sc_y,    0, 0],
                         [   0,     0, sc_z, 0],
                         [   0,     0,    0, 1]])

        # get the tan of angles
        tan_shx = np.tan(theta_shx)
        tan_shy = np.tan(theta_shy)
        tan_shz = np.tan(theta_shz)
        # get the shearing matrix on x axis
        Sh_Mx = np.array([[      1, 0, 0, 0],
                          [tan_shy, 1, 0, 0],
                          [tan_shz, 0, 1, 0],
                          [      0, 0, 0, 1]])
        # get the shearing matrix on y axis
        Sh_My = np.array([[1, tan_shx, 0, 0],
                          [0,       1, 0, 0],
                          [0, tan_shz, 1, 0],
                          [0,       0, 0, 1]])
        # get the shearing matrix on z axis
        Sh_Mz = np.array([[1, 0, tan_shx, 0],
                          [0, 1, tan_shy, 0],
                          [0, 0,       1, 0],
                          [0, 0,       0, 1]])
        # compute the full shearing matrix
        Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

        Identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # compute the full transform matrix
        M = Identity
        M = np.dot(Sh_M, M)
        M = np.dot(R_M,  M)
        M = np.dot(Sc_M, M)
        M = np.dot(T_M,  M)
        M = np.dot(Hp_M, np.dot(M, H_M))
        # apply the transformation
        X = cv2.warpPerspective(X, M, (w, h))
        
        if type(Y) == type(None):
            return X
        else:
            Y = cv2.warpPerspective(Y, M, (w, h))
            return X,Y
        
class Contrast(object):
    "Random Contrast"
    
    def __init__(self, range_contrast:Tuple[int,int]=(-50, 50)):
        self.range_contrast = range_contrast

    def __call__(self, X:np.ndarray, Y:Union[np.ndarray,None])-> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply random contrast
        
        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image, not modified
        '''
        contrast = np.random.randint(*self.range_contrast)
        X = X * (contrast / 127 + 1) - contrast

        if type(Y) != type(None):
            return X, Y
        else:
            return X
    
class UniformNoise(object):
    "Uniform Random Noise"
    def __init__(self, low:float=-50, high:float=50):
        self.low = low
        self.high = high

    def __call__(self, X:np.ndarray, Y:Union[np.ndarray,None]) -> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply a random uniform noise.
        
        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image, not modified
        '''
        noise = np.random.uniform(self.low, self.high, X.shape)
        X = X + noise
        
        if type(Y) != type(None):
            return X, Y
        else:
            return X
    
class GaussianNoise(object):
    "Random Gaussian Noise"
    def __init__(self, center=0, std=50):
        self.center = center
        self.std = std

    def __call__(self, X, Y)-> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply a random Gaussian noise
        
        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image, not modified
        '''
        noise = np.random.normal(self.center, self.std, X.shape)
        X = X + noise
        
        if type(Y) != type(None):
            return X, Y
        else:
            return X
    
class Vignetting(object):
    "Random vignetting"
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X, Y)-> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply a random Vignetting
        
        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image
        '''
        h, w = X.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X = X * (1 + sign * vignette)

        if type(Y) != type(None):
            return X, Y
        else:
            return X

class LensDistortion(object):
    "Lens distorsion"
    def __init__(self, d_coef=(0.15, 0.15, 0.1, 0.1, 0.05)):
        self.d_coef = np.array(d_coef)

    def __call__(self, X, Y)-> Union[np.ndarray,Tuple[np.ndarray]]:
        '''
        Apply lens distortion

        Params
        ---------
            X: np.ndarray
                original image
            Y=None: np.ndarray
                mask image
        '''

        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0,     1]])

        d_coef = self.d_coef * np.random.random(5) # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1) # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        X = cv2.remap(X, *remap, cv2.INTER_LINEAR)
        Y = cv2.remap(Y, *remap, cv2.INTER_LINEAR)
        
        
        if type(Y) != type(None):
            return X, Y
        else:
            return X