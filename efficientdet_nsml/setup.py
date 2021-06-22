##nsml: nvcr.io/nvidia/pytorch:20.03-py3

from distutils.core import setup

setup(
    name='nsml test example',
    version='1.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'tqdm',
        'tensorboard',
        'tensorboardX',
        'pyyaml',
        'webcolors',
        'Pillow',
        'opencv-python',
        'natsort'
    ]
)