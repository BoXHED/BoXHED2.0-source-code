from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="boxhed",
    version="2.0.4",
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="BoXHED2.0",
    author='Arash Pakbin',
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_package_data = True,
    zip_safe=False,
    install_requires = ['pandas', 'scikit-learn', 'matplotlib', 'tqdm', 'py3nvml', 'boxhed_kernel', 'boxhed_prep']
)