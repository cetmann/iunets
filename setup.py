import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

VERSION = 0.1
DESCRIPTION = "Invertible U-Nets for memory efficiency in Pytorch"
LONG_DESCRIPTION = "A package which allows for training segmentation models "\
                   "in Pytorch for high-dimensional 3D data, using invertible "\
                   "U-Nets (iUNets)."


setup(
    name="iunets",
    version=VERSION,
    author="Christian Etmann",
    author_email="cetmann@damtp.cam.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['memcnn'],
    keywords=['python', 'neural network', 'segmentation', 'unet', 'u-net',
              'invertible', 'medical imaging'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)