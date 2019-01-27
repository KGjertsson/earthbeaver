from distutils.core import setup

INSTALL_REQUIRES = ['numpy', 'pandas', 'keras==2.1.1', 'matplotlib', 'tqdm',
                    'scikit-learn', 'obspy', 'lightgbm', 'xgboost', 'catboost']

try:
    import tensorflow
except ImportError:
    INSTALL_REQUIRES += ['tensorflow']

setup(
    name='earthbeaver',
    version='0.1.0',
    packages=['beaver'],
    install_requires=INSTALL_REQUIRES
)
