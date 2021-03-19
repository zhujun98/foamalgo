Building the Documentation
==========================

Prerequisites
-------------

The documentation build process uses `Doxygen <http://www.doxygen.nl/>`_,
`Sphinx <http://www.sphinx-doc.org/>`_ and `Breathe <https://breathe.readthedocs.io>`_
along with a few extensions.

.. code-block:: shell

    conda install -c conda-forge doxygen
    conda install -c conda-forge --file docs/requirements.txt


Build
-----

Process the C++ API using Doxygen.

.. code-block:: shell

    doxygen Dockfile

Build the complete documentation using Sphinx.

.. code-block:: shell

    make html
