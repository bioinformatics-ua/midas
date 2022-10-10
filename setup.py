from setuptools import find_packages, setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
import midas
install_reqs = parse_requirements("requirements.txt", session=False)

# https://stackoverflow.com/questions/62114945/attributeerror-parsedrequirement-object-has-no-attribute-req
#requirements = list(requirements) 
try:
    requirements = [str(ir.req) for ir in install_reqs]
except:
    requirements = [str(ir.requirement) for ir in install_reqs]

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
