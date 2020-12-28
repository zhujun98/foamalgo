foamalgo
========

[![Lates Release](https://img.shields.io/github/v/release/zhujun98/foamalgo)](https://github.com/zhujun98/foamalgo/releases)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/zhujun98/foamalgo.svg?branch=master)](https://travis-ci.com/zhujun98/foamalgo)
[![Build Status](https://dev.azure.com/zhujun981661/zhujun981661/_apis/build/status/zhujun98.foamalgo?branchName=master)](https://dev.azure.com/zhujun981661/zhujun981661/_build/latest?definitionId=2&branchName=master)
[![Documentation](https://img.shields.io/readthedocs/foamalgo)](https://foamalgo.readthedocs.io/en/latest/)

![Language](https://img.shields.io/badge/language-c++-red)
![Language](https://img.shields.io/badge/language-python-blue)


## Introduction

`foamalgo` is a head-only C++ (17) library with Python binding `pyfoamalgo`, 
which is meant to be used in numerical analysis in photon science and 
accelerator physics. It leverages SIMD and multi-threaded parallelism
to increase algorithm performance by up to 2 orders of magnitude compared to 
the implementations in other libraries like [numpy](https://numpy.org/).

`foamalgo` has been developed based on the algorithm package in [EXtra-foam](https://github.com/European-XFEL/EXtra-foam),
which is the framework for real-time (online) data analysis and visualization 
of big data from various large detectors at European XFEL and has stood 
the test of 24/7 operations. By separating the algorithm code in a 
stand-alone library, it facilitates the maintanance of the large code base
and benefits the offline analysis as well. 

## Installation

Install the Python package

```sh
conda install -c conda-forge pyfoamalgo
```

Install the head-only C++ library

```sh
conda install -c conda-forge foamalgo
```

## Documentation

For more information on `foamalgo` and `pyfoamalgo`, please checkout the documentation

https://foamalgo.readthedocs.io/en/latest/index.html
