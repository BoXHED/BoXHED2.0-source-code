from asyncio import subprocess
import os
import sys
import shutil
import subprocess
from setuptools import setup, find_packages
'''
#https://stackoverflow.com/questions/19569557/pip-not-picking-up-a-custom-install-cmdclass
from setuptools.command.install  import install
from setuptools.command.build    import build
from setuptools.command.develop  import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist    import sdist

def cmake_args():
    if sys.platform == 'win32':
        return ["-GVisual Studio 17 2022", "-Ax64"]
    return []

def cmake_build_args():
    
    if sys.platform == 'win32':
        target = 'ALL_BUILD'
    elif sys.platform.startswith('darwin') or sys.platform.startswith('linux') or sys.platform.startswith('freebsd'):
        target = 'all'
    else:
        raise OSError("ERROR: platform not supported!")
    return ['--target'+target]


def run_cmake():
    
    print (["cmake", '..']           + cmake_args())
    print (["cmake", '--build', '.'] + cmake_build_args())
    #return
    build_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build')
    #print (build_dir)
    #if os.path.exists(build_dir):
    #    shutil.rmtree(build_dir)
    #os.mkdir(build_dir)

    if not os.path.exists(build_dir):
    #    shutil.rmtree(build_dir)
        os.mkdir(build_dir)

    #subprocess.run(["cmake", '..']           + cmake_args(),       cwd = build_dir, shell=True)
    #subprocess.run(["cmake", '--build', '.'] + cmake_build_args(), cwd = build_dir, shell=True)

    subprocess.run(["cmake", '..']           ,       cwd = build_dir)
    subprocess.run(["cmake", '--build', '.'] , cwd = build_dir)

import sys
class RunSetupsInstall(install):
    def run(self):
        install.run(self)
        run_cmake()

class RunSetupsBuild(build):
    def run(self):
        build.run(self)
        run_cmake()


class RunSetupsDevelop(develop):
    def run(self):
        develop.run(self)
        run_cmake()


class RunSetupsEgg_info(egg_info):
    def run(self):
        egg_info.run(self)
        run_cmake()

class RunSetupsSdist(sdist):
    def run(self):
        sdist.run(self)
        run_cmake()
'''

setup(
    name="boxhed_prep",
    version="2.2",
    description="preprocessor for BoXHED2.0",
    author='Arash Pakbin',
    #cmake_install_target = "ll",#cmake_install_target(),
    #cmake_args=cmake_args(),
    #install_requires=[
    #'numpy',
    #'scipy',
    #],
    #license="MIT",
    #cmdclass={
    #    'install':  RunSetupsInstall,
    #    'build':    RunSetupsBuild,
    #    'develop':  RunSetupsDevelop,
    #    'egg_info': RunSetupsEgg_info,
    #    'sdist':    RunSetupsSdist
    #},

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