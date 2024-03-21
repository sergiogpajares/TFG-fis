# Open CV 4.9.0 Ubuntu installation guide

## nvidia driver

The driver `nvidia-driver-535` comes preinstalled with Ubuntu 22.04. We will stick to that one. If we want to check weather everything is running fine you can execute `nvidia-smi` and you'll get something similar to the following output
```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1050        Off | 00000000:01:00.0 Off |                  N/A |
| N/A   35C    P8              N/A / ERR! |      4MiB /  4096MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1694      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

Additional drivers can be install with apt if the proper keyrings are installed. No compilation is required!

## General dependencies

You'll need general compilation tools such as,

```bash
sudo apt-get install cmake build-essential
```

Then, the following libraries are requiered
```bash
sudo apt-get install unzip pkg-config zlib1g -y
sudo apt-get install libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev -y
sudo apt-get install libpng-dev libtiff-dev libglew-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev -y
sudo apt-get install libgtk2.0-dev libgtk-3-dev libcanberra-gtk* -y
sudo apt-get install python-dev python-numpy python-pip -y
sudo apt-get install python3-dev python3-numpy python3-pip -y
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev -y
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev -y
sudo apt-get install gstreamer1.0-tools libgstreamer-plugins-base1.0-dev -y
sudo apt-get install libgstreamer-plugins-good1.0-dev -y
sudo apt-get install libv4l-dev v4l-utils v4l2ucp qv4l2 -y
sudo apt-get install libtesseract-dev libxine2-dev libpostproc-dev -y
sudo apt-get install libavresample-dev libvorbis-dev -y
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev -y
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev -y
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev -y
sudo apt-get install liblapack-dev liblapacke-dev libeigen3-dev gfortran -y
sudo apt-get install libhdf5-dev libprotobuf-dev protobuf-compiler -y
sudo apt-get install libgoogle-glog-dev libgflags-dev -y
```

## CUDA Toolkit

It can be downloaded form [https://developer.nvidia.com/cuda-12-2-0-download-archive](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)

For Ubuntu 22.04 x86_64 you have the following steps
```bash 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

If anything goes wrong you have to uninstalled the packages  to avoid conflits. Remeber to use the `--purge` flag:
```bash
sudo apt-get remove --purge <package>
```

The following command is useful to remove `residual-config` files from deleted packages.
```bash
$ sudo apt-get clean && sudo apt-get purge -y $(dpkg -l | grep '^rc' | awk '{print $2}')
```

## CuDNN

As I'm wirtting this `CuDNN 9.0.0` is not supported. Some compilation errors will occur. An older version must be choosen. In my case the oldest available for `CUDA 12.x`. Going to [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the one you need. In my case [this one](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/12.0/cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb/) **was proven unsuccesful**

Instead use
```bash
sudo apt-get install libcudnn8-dev
```

> warning:
> I got some question about how I managed to get it working. I had to include
`cmake -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.8 -DCUDNN_INCLUDE_DIR=/usr/include/ ..`

## Nvidia Video codec SDK

```bash
sudo cp ~/Downloads/Video_Codec_SDK_12.1.14/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/cuda-12.2/lib64/
sudo cp ~/Downloads/Video_Codec_SDK_12.1.14/Lib/linux/stubs/x86_64/libnvidia-encode.so /usr/local/cuda-12.2/lib64/
sudo cp ~/Downloads/Video_Codec_SDK_12.1.14/Interface/cuviddec.h /usr/local/cuda-12.2/include/
sudo cp ~/Downloads/Video_Codec_SDK_12.1.14/Interface/nvcuvid.h  /usr/local/cuda-12.2/include/
```

## Compilation procesess

Two repositories are required to get `CUDA` support. The main repo is `opencv` but all the `CUDA` accelerated runtimes are in `opencv_contrib` repo. Generally, cloning the last commit will be enough. However, if you want to check out to a specific tag you should remove the `--depth=1`. Remember that you can always get some older commits by applying `git fetch --depth=<number of commits>`. 

```bash
git clone https://github.com/opencv/opencv.git --depth=1
git clone https://github.com/opencv/opencv_contrib.git --depth=1
```

Move into `opencv` and create a directory `build`, then move into `build`.

Project is configured with `cmake`. The following flags are used:

```bash
cmake .. \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/GitRepositories/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=OFF \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_WEBP=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=OFF \
-D BUILD_TBB=OFF \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_PROTOBUF=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_opencv_python2=OFF \
-D BUILD_JAVA=OFF
```
Even tought I got `TBB` install as a general dependency, I had to remove it due to an error.

If you get an error about `cudnn` you may need to manually specify the path:
```bash
cmake -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.8 -DCUDNN_INCLUDE_DIR=/usr/include/ ..
```
Check your `CuDNN` path using
```bash
find cudnn /usr/ | grep cudnn
```

After configuring the project run `make`. Remember to use the `-j <number of threads>`. Yo can check how many threads you can run in parallel using `lscpu`. In case 12:
```bash
make -j12
```
The building process is quite slow and it can take up to 8 hours to complete. However, on a very fast computer you can expect a compiling time of ~1 h. Some slower PC may take 2-4 hours.

> warning: **Make sure you got enough memory**
> At the end of the process, when building the python binding a huge amout of RAM memory is requiered, about 4 Gb. If you get a compiler error it may be due to a runout of memory in the system. If this happens you must enlarge (or create) your swap memory. If you're running a Jetson Nano you'll run out of memory.

We will move into the folder `python_loader` that will contain
1. `cv2` directory
2. `setup.py`

Then we can build a python wheel to be installed in our virtual enviroments. Note that `setuptools>=40.0.0` is requiered.

```bash
python3 -m build --wheel
```

In the directory `dist` you'll find the `opencv-4.9.0-py3-none-any.whl` wheel file.

## Check installation

Running python you can print the build information using 

```python
import cv2
print( cv2.getBuildInformation() )
```

You'll get something similar to:
```bash

General configuration for OpenCV 4.9.0 =====================================
  Version control:               4.9.0

  Extra modules:
    Location (extra):            /home/sergio/GitRepositories/opencv_contrib/modules
    Version control (extra):     bf95e79

  Platform:
    Timestamp:                   2024-03-20T18:46:24Z
    Host:                        Linux 6.5.0-26-generic x86_64
    CMake:                       3.22.1
    CMake generator:             Unix Makefiles
    CMake build tool:            /usr/bin/gmake
    Configuration:               RELEASE

  CPU/HW features:
    Baseline:                    SSE SSE2 SSE3
      requested:                 SSE3
      disabled:                  NEON
    Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
      requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
      SSE4_1 (16 files):         + SSSE3 SSE4_1
      SSE4_2 (1 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
      FP16 (0 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
      AVX (8 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
      AVX2 (36 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
      AVX512_SKX (5 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_COMMON AVX512_SKX

  C/C++:
    Built as dynamic libs?:      YES
    C++ standard:                11
    C++ Compiler:                /usr/bin/c++  (ver 11.4.0)
    C++ flags (Release):         -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
    C++ flags (Debug):           -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
    C Compiler:                  /usr/bin/cc
    C flags (Release):           -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG
    C flags (Debug):             -fsigned-char -ffast-math -fno-finite-math-only -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG
    Linker flags (Release):      -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
    Linker flags (Debug):        -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
    ccache:                      YES
    Precompiled headers:         NO
    Extra dependencies:          m pthread cudart_static dl rt nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cudnn cufft -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu
    3rdparty dependencies:

  OpenCV modules:
    To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hdf hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor mcc ml objdetect optflow phase_unwrapping photo plot python3 quality rapid reg rgbd saliency sfm shape signal stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto
    Disabled:                    world
    Disabled by dependency:      -
    Unavailable:                 cannops cvv java julia matlab ovis python2 viz
    Applications:                perf_tests apps
    Documentation:               NO
    Non-free algorithms:         YES

  GUI:                           GTK3
    GTK+:                        YES (ver 3.24.33)
      GThread :                  YES (ver 2.72.4)
      GtkGlExt:                  NO
    VTK support:                 NO

  Media I/O: 
    ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.11)
    JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver 80)
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.6.37)
    TIFF:                        build (ver 42 - 4.2.0)
    JPEG 2000:                   build (ver 2.5.0)
    OpenEXR:                     build (ver 2.3.0)
    HDR:                         YES
    SUNRASTER:                   YES
    PXM:                         YES
    PFM:                         YES

  Video I/O:
    DC1394:                      NO
    FFMPEG:                      YES
      avcodec:                   YES (58.134.100)
      avformat:                  YES (58.76.100)
      avutil:                    YES (56.70.100)
      swscale:                   YES (5.9.100)
      avresample:                NO
    GStreamer:                   YES (1.20.3)
    v4l/v4l2:                    YES (linux/videodev2.h)

  Parallel framework:            OpenMP

  Trace:                         YES (with Intel ITT)

  Other third-party libraries:
    Intel IPP:                   2021.10.0 [2021.10.0]
           at:                   /home/sergio/GitRepositories/opencv/build/3rdparty/ippicv/ippicv_lnx/icv
    Intel IPP IW:                sources (2021.10.0)
              at:                /home/sergio/GitRepositories/opencv/build/3rdparty/ippicv/ippicv_lnx/iw
    VA:                          NO
    Lapack:                      YES (/usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libcblas.so /usr/lib/x86_64-linux-gnu/libatlas.so)
    Eigen:                       YES (ver 3.4.0)
    Custom HAL:                  NO
    Protobuf:                    build (3.19.1)
    Flatbuffers:                 builtin/3rdparty (23.5.9)

  NVIDIA CUDA:                   YES (ver 12.2, CUFFT CUBLAS NVCUVID FAST_MATH)
    NVIDIA GPU arch:             50 52 60 61 70 75 80 86 89 90
    NVIDIA PTX archs:            90

  cuDNN:                         YES (ver 8.9.7)

  Python 3:
    Interpreter:                 /usr/bin/python3 (ver 3.10.12)
    Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.10.so (ver 3.10.12)
    numpy:                       /usr/lib/python3/dist-packages/numpy/core/include (ver 1.21.5)
    install path:                /usr/lib/python3/dist-packages/cv2/python-3.10

  Python (for build):            /usr/bin/python3

  Install to:                    /usr
```

Make sure you got `CuDNN` support!