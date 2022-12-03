from setuptools import setup


setup(
    name="boxhed_prep",
    version="2.42",
    description="preprocessor for BoXHED2.0",
    author='Arash Pakbin',

    install_requires=[
            'cmake',
            'numpy',
            'pandas'
        ],
    packages=['boxhed_prep'],
    python_requires=">=3.8",
    include_package_data=True,
    install_package_data = True,
    zip_safe=False
)