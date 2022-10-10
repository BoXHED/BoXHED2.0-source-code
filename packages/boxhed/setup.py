import sys
from setuptools import find_packages, setup  # This line replaces 'from setuptools import setup'


setup(
    name="boxhed",
    version="2.0",
    description="BoXHED2.0",
    author='Arash Pakbin',
    #cmake_install_target = cmake_install_target(),
    #cmake_args=cmake_args(),
    #install_requires=[
    #'numpy',
    #'scipy',
    #],
    #license="MIT",
    packages=find_packages(),#['preprocessor'],
    #py_modules=["boxhed/boxhed"],
    python_requires=">=3.8",
    include_package_data=True,
    install_package_data = True,
    zip_safe=False,
    install_requires = ['pandas', 'scikit-learn', 'matplotlib', 'tqdm', 'py3nvml', 'boxhed_kernel', 'boxhed_prep']
)