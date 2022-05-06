from setuptools import setup, find_packages

setup(
    name='retrain_ablooper',
    version='1.0',
    description='Set of functions to retrain ABlooper on IMGT numbering scheme',
    maintainer='Fabian Spoendlin',
    maintainer_email='fabian.spoendlin@exeter.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('retrain_ablooper', 'retrain_ablooper.*', 'ABlooper', 'ABlooper.*')),
    install_requires=[
        'numpy',
        'einops>=0.3',
        'torch>=1.6',
        'rich',
        'pandas',
        'scikit-learn',
    ],
)
