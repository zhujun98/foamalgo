foamalgo
========

[![Lates Release](https://img.shields.io/github/v/release/zhujun98/foamalgo)](https://github.com/zhujun98/foamalgo/releases)
[![License](https://img.shields.io/github/license/zhujun98/foamalgo)](https://github.com/zhujun98/foamalgo/releases)
[![Build Status](https://travis-ci.org/zhujun98/foamalgo.svg?branch=master)](https://travis-ci.org/zhujun98/foamalgo)
[![Documentation](https://img.shields.io/readthedocs/foamalgo)](https://foamalgo.readthedocs.io/en/latest/)
[![Documentation](https://img.shields.io/badge/documentation-online-blue)](https://foamalgo.readthedocs.io/en/latest/)

![Language](https://img.shields.io/badge/language-c++-red)
![Language](https://img.shields.io/badge/language-python-blue)


## Installation

```shell script
$ git clone --recursive https://github.com/zhujun98/foamalgo.git
# in case submodules were not cloned by including '--recursive'
$ git submodule update --init

$ cd foamalgo
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix 
$ make install
```

#### pyfoamalgo

```shell script
$ cd foamalgo
$ pip install . -v
```
