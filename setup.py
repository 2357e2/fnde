from setuptools import setup, find_packages
setup(
    name='fnde',
    version='0.1.0',
    packages=find_packages(include=['fnde', 'fnde.*', 'utils', 'models', 'data_handling', 'tests', 'training']),
    install_requires=[]
)