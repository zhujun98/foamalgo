Installation
============

Using the conda package
-----------------------

foamalgo
""""""""

.. code:: shell

    conda install -c conda-forge foamalgo

pyfoamalgo
""""""""""

.. code:: shell

    conda install -c conda-forge pyfoamalgo


From source
-----------

`foamalgo` requires a modern C++ compiler which supports C++17.

.. code:: shell

    conda env update -f environment-dev.yml
    git clone https://github.com/zhujun98/foamalgo.git
    cd foamalgo

foamalgo
""""""""

.. code:: shell

    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix
    make install

pyfoamalgo
""""""""""

.. code:: shell

    export CC=gcc-9 CXX=g++-9  # Specify the compiler in your system which supports C++17
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    pip install . -v


Bugs and workarounds
"""""""""""""""""""""""

- There is a compiler bug related to GCC-7: https://github.com/xtensor-stack/xtensor/issues/2289.

- Install `h5py` on MacOS with m1 chip.

.. code:: shell

    brew install hdf5
    export HDF5_DIR=/opt/homebrew/Cellar/hdf5/<version>
    pip install --no-binary=h5py h5py


On the Maxwell cluster
----------------------

.. code:: shell

    module load anaconda3

    # It is highly recommended to create an independent environment.
    conda create -n foamalgo python=3.7
    conda activate foamalgo

    conda install -c conda-forge pyfoamalgo
    conda install jupyter
    python -m ipykernel install --user --name foamalgo --display-name "foamalgo"

Now, you should be able to start a Jupyter notebook via https://max-jhub.desy.de/ and
select the newly created kernel.
