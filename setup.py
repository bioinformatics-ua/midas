from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

import midas

setup(
    name='MultI-framework DataloAderS',
    packages=find_packages(include=['midas*']),
    version=midas.__version__,
    description='A multi-framework dataloaders powered by tensorflow.data.Dataset API',
    author='Tiago Almeida',
    author_email='tiagomeloalmeida@ua.pt',
    license='Apache License 2.0',
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest','tensorflow','jax','torch'],
    test_suite='tests',
)
