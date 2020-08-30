Installation
============


From source
-----------

foamalgo
""""""""

.. code::

    git clone --recursive https://github.com/zhujun98/foamalgo.git
    # in case submodules were not cloned by including '--recursive'
    git submodule update --init

    cd foamalgo
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix
    make install


pyfoamalgo
""""""""""

.. code::

    git clone --recursive https://github.com/zhujun98/foamalgo.git

    cd foamalgo
    pip install . -v
