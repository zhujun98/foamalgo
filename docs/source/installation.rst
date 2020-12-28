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

`foamalgo` requires a modern C++ compiler which supports C++17.

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
    export CC=gcc-7 GXX=g++-7  # Specify the compiler in your system which supports C++17
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    pip install . -v


On the Maxwell cluster
----------------------

.. code::

    module load anaconda3

    # It is highly recommended to create an independent environment.
    conda create -n foam python=3.7
    conda activate foam

    conda install -c conda-forge pyfoamalgo
    conda install jupyter
    python -m ipykernel install --user --name foam --display-name "foam"

Now, you should be able to start a Jupyter notebook via https://max-jhub.desy.de/ and
select the newly created kernel.
