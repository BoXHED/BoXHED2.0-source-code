from setuptools import find_packages, setup


setup(
    name="boxhed",
    version="2.4",
    description="BoXHED2.0",
    author='Arash Pakbin',
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_package_data = True,
    zip_safe=False,
    install_requires = ['pandas', 'scikit-learn', 'matplotlib', 'tqdm', 'py3nvml', 'boxhed_kernel', 'boxhed_prep']
)