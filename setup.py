from setuptools import setup, find_packages


setup(
    name='r2d2',
    version='1.0',
    packages=find_packages(include=['r2d2/*']),
    include_package_data=True,
    package_data={'r2d2.models': ['*.*'] ,'r2d2.results': ['*.*']},

    # Declare your packages' dependencies here, for eg:
    install_requires=['tqdm', 'pillow', 'numpy', 'scipy', 'matplotlib',  # from conda regular channel
                      # 'pytorch', 'torchvision', 'cudatoolkit',  # from conda pytorch channel, don't seem to work with pip
                      ],

    author='NAVER Corp., Jérome Revaud',
    author_email='jerome.revaud@naverlabs.com',

    summary='R2D2: Reliable and Repeatable Detector and Descriptor',
    url='https://github.com/naver/r2d2',
    license='Creative Commons BY-NC-SA 3.0',
)
