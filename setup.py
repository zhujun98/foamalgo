"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""
import contextlib
import glob
import multiprocessing as mp
import os
import os.path as osp
import re
import shutil
import sys
import subprocess
from setuptools import setup, Command, find_packages, Distribution, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.version import LooseVersion
from distutils.util import strtobool


@contextlib.contextmanager
def changed_cwd(dirname):
    oldcwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(oldcwd)


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=''):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


ext_modules = [
    CMakeExtension("pyfoamalgo"),
]


class Build(build_ext):

    description = "Build the C++ extensions for pyfoamalgo"
    user_options = [
        ('disable-tbb', None, 'disable oneTBB'),
        ('disable-xsimd', None, 'disable XSIMD'),
        ('with-tests', None, 'build cpp unittests'),
    ] + build_ext.user_options

    def initialize_options(self):
        super().initialize_options()

        self.disable_tbb = strtobool(os.environ.get('DISABLE_TBB', '0'))
        self.disable_xsimd = strtobool(os.environ.get('DISABLE_XSIMD', '0'))
        self.with_tests = strtobool(os.environ.get('BUILD_FOAMALGO_CPP_TESTS', '0'))

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the "
                               "following extensions: " + ", ".join(
                e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        cmake_minimum_version_required = '3.15.0'
        if cmake_version < cmake_minimum_version_required:
            raise RuntimeError(f"CMake >= {cmake_minimum_version_required} "
                               f"is required!")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        build_type = 'debug' if self.debug else 'release'
        lib_output_dir = osp.join(
            osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name))), "pyfoamalgo/lib")

        cmake_options = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_output_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_PREFIX_PATH={os.getenv('CMAKE_PREFIX_PATH')}",
            f"-DBUILD_PYFOAMALGO=ON",
        ]

        def _opt_switch(x):
            return 'ON' if x else 'OFF'

        cmake_options.append(
            f'-DFOAMALGO_USE_TBB={_opt_switch(not self.disable_tbb)}')
        cmake_options.append(
            f'-DXTENSOR_USE_TBB={_opt_switch(not self.disable_tbb)}')

        cmake_options.append(
            f'-DFOAMALGO_USE_XSIMD={_opt_switch(not self.disable_xsimd)}')
        cmake_options.append(
            f'-DXTENSOR_USE_XSIMD={_opt_switch(not self.disable_xsimd)}')

        cmake_options.append(
            f'-DBUILD_FOAMALGO_CPP_TESTS={_opt_switch(self.with_tests)}')

        max_jobs = os.environ.get('BUILD_FOAM_MAX_JOBS', str(mp.cpu_count()))
        build_options = ['--', '-j', max_jobs]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        with changed_cwd(self.build_temp):
            # generate build files
            print("-- Running cmake for foamalgo")
            self.spawn(['cmake', ext.source_dir] + cmake_options)
            print("-- Finished cmake for foamalgo")

            # build
            print("-- Running cmake --build for foamalgo")
            self.spawn(['cmake', '--build', '.'] + build_options)
            print("-- Finished cmake --build for foamalgo")


class Test(build_ext):
    def run(self):
        # build and run cpp test
        with changed_cwd(self.build_temp):
            self.spawn(['make', 'ftest'])

        # run Python test
        import pytest
        errno = pytest.main(['pyfoamalgo'])
        sys.exit(errno)


class Benchmark(Command):

    user_options = []

    def initialize_options(self):
        """Override."""
        pass

    def finalize_options(self):
        """Override."""
        pass

    def run(self):
        self.spawn(['python', 'benchmarks/benchmark_imageproc.py'])
        self.spawn(['python', 'benchmarks/benchmark_geometry.py'])
        self.spawn(['python', 'benchmarks/benchmark_statistics.py'])


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


this_directory = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(this_directory, 'pyfoamalgo/version.py')

try:
    exec(open(version_file).read())
except IOError:
    print(f"Failed to load pyfoamalgo version file for packaging. " +
          f"'{version_file}' not found!")
    sys.exit(-1)

VERSION = __version__

setup(
    name='pyfoamalgo',
    version=VERSION,
    author='Jun Zhu',
    author_email='zhujun981661@gmail.com',
    description='',
    long_description='',
    url='',
    packages=find_packages(),
    ext_modules=ext_modules,
    tests_require=['pytest'],
    cmdclass={
        'clean': clean,
        'build_ext': Build,
        'test': Test,
        'benchmark': Benchmark,
    },
    distclass=BinaryDistribution,
    package_data={
        'pyfoamalgo': [
            'geometry/*.h5',
        ]
    },
    install_requires=[
        'numpy>=1.16.1',
        'scipy>=1.2.1',
        'h5py>=2.10.0',
    ],
    extras_require={
        'test': [
            'extra-geom==1.6.0',
            'pytest',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
