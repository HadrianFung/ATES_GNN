from setuptools import setup, find_packages
import os

# Utility function to read the README file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Read requirements from requirements.txt
with open('requirements.txt') as req_file:
    pack = req_file.readlines()

setup(
    name="well_preprocessor",
    version="0.1.0",
    description="A package for well preprocessing and visualization",
    long_description=read('README.md'),
    author="Hadrian Fung",
    author_email="hadrian.fung23@imperial.ac.uk",
    url="https://github.com/ese-msc-2023/irp-hf923.git", 
    packages=find_packages(),
    include_package_data=True,
    install_requires=pack,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)