Welcome to foamalgo's documentation!
====================================

Introduction
------------

`foamalgo` is a head-only C++ library with Python binding `pyfoamalgo`,
which is meant to be used in numerical analysis in photon science and
accelerator physics. It leverages SIMD and multi-threaded parallelism
to increase algorithm performance by up to 2 orders of magnitude compared to
the implementations in other libraries like `numpy <https://numpy.org/>`_.

`foamalgo` has been developed based on the algorithm package in `EXtra-foam <https://github.com/European-XFEL/EXtra-foam>`_,
which is the framework for real-time (online) data analysis and visualization
of big data from various large detectors at European XFEL and has stood
the test of 24/7 operations. By separating the algorithm code in a
stand-alone library, it facilitates the maintanance of the large code base
and benefits the offline analysis as well.

.. toctree::
   :caption: INSTALLATION:
   :maxdepth: 1

   installation
   changelog

.. toctree::
   :caption: EXAMPLES
   :maxdepth: 2

   C++ <cpp/examples>
   Python <python/examples>

.. toctree::
   :maxdepth: 2
   :caption: API REFERENCE

   C++ <cpp/index>
   Python <python/index>

.. toctree::
   :maxdepth: 1
   :caption: DEVELOPMENT
