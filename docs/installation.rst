Installation
============

Using the conda package
-----------------------

foamalgo
""""""""

.. code::

    conda install -c conda-forge foamalgo

pyfoamalgo
""""""""""

.. code::

    conda install -c conda-forge pyfoamalgo


From source
-----------

foamalgo
""""""""

.. code::

    conda install -c anaconda cmake
    conda install -c conda-forge tbb-devel xsimd xtensor xtensor-blas

    git clone https://github.com/zhujun98/foamalgo.git

    cd foamalgo
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix
    make install

pyfoamalgo
""""""""""

.. code::

    conda install -c anaconda cmake
    conda install -c conda-forge tbb-devel xsimd xtensor xtensor-blas numpy xtensor-python

    git clone https://github.com/zhujun98/foamalgo.git

    cd foamalgo
    pip install . -v
