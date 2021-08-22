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

    conda create --yes --quiet --name foamalgo --python=3.7
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

.. warning::

    There is a compiler bug related to GCC-7: https://github.com/xtensor-stack/xtensor/issues/2289.


On the Maxwell cluster
----------------------

.. code:: shell

    module load anaconda3

    # It is highly recommended to create an independent environment.
    conda create -n foam python=3.7
    conda activate foam

    conda install -c conda-forge pyfoamalgo
    conda install jupyter
    python -m ipykernel install --user --name foam --display-name "foam"

Now, you should be able to start a Jupyter notebook via https://max-jhub.desy.de/ and
select the newly created kernel.
