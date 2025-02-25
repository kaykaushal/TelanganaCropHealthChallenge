

.. _ENVIRONMENT_MAIN:

***********************************
Using the Project's environment
***********************************

A short description of the project.

**Author**: Kaushal K (`Your email address <mailto:Your email address>`_)

.. _env_install_subsec:

Installing Environment & Dependencies
=====================================

To use the scripts in this repository, you must have `Anaconda <https://www.anaconda.com/download/#macos>`_ installed on the systems that will
be running the scripts. This will simplify the processes of installing 
all the dependencies.

For reference, see: `Manage Anaconda Environments <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ 

The package counts with a **Makefile** with useful commands and functions.
You must use this Makefile to ensure that you have all of the necessary 
*dependencies*, as well the correct **conda environment**.

.. _env_makefile_funcs:

Show all available functions in the Makefile
--------------------------------------------

You can use the *Makefile* for running common tasks like 
*updating environments*, *cleaning extra files*, and more.

To show all available functions in the Makefile, run:

.. code-block:: text

    make show-help

    Available rules:

    clean               Deletes all build, test, coverage, and Python artifacts
    clean-build         Remove build artifacts
    clean-pyc           Removes Python file artifacts
    clean-test          Remove test and coverage artifacts
    environment         Set up python interpreter environment - Using environment.yml
    lint                Lint using flake8
    remove_environment  Delete python interpreter environment
    test_environment    Test python environment is setup correctly
    update_environment  Update python interpreter environment

.. _create_env:

Create environment
-------------------

In order to properly run the commands of this project, you should install the 
**necessary packages** before. For this, you will to have installed 
**Anaconda**, because otherwise you will not be able to use this command.

The name of the environment and its dependencies are explicitely shown in the 
``environment.yml`` file.
In order to create the environment, you must run:

.. code-block:: text

    make environment

The main file that lists all of the dependencies for the project can 
be found as ``environment.yml``.

.. _activate_env:

Activating the environment
----------------------------

Once the environment has been **installed**, you can now *activate* the 
environment by typing

.. code-block:: text

    source activate geoai

.. note::

    Depending on your installation of Anaconda, you might have to use the 
    command: 

    .. code-block:: text
    
        conda activate geoai

    instead.

.. _updating_env:

Updating environment
--------------------

You can always update the project's environment. The package dependencies
are handled by the ``environment.yml`` file, and sometimes these packages 
need to updaetd.

You can updated the project's environments by running:

.. code-block:: text

    make update_environment

This will update the versions of each of the necessary packages.

.. _deactivating_env:

Deactivating environment
-------------------------

Once you are done running the scripts of this project, you should 
**deactivate** the environment. To do so, run:

.. code-block:: text

    source deactivate

.. note::

    Depending on your installation of Anaconda, you might have to use the 
    command: 

    .. code-block:: text
    
        conda deactivate

    instead.

.. _auto_activate_env:

Auto-activate environment
-------------------------

To make it easier to activate the necessary environment, one can use the 
`conda-auto-env <https://github.com/chdoig/conda-auto-env>`_ package,
which **activates** the necessary environment **automatically**.

See the link above for more information!






.. ----------------------------------------------------------------------------

Project based on the `modified <https://github.com/vcalderon2009/cookiecutter-data-science-vc>`_  version of
`cookiecutter data science project template <https://drivendata.github.io/cookiecutter-data-science/>`_ 

.. |Issues| image:: https://img.shields.io/github/issues/vcalderon2009/GeoAI2025.svg
   :target: https://github.com/vcalderon2009/GeoAI2025/issues
   :alt: Open Issues

.. |RTD| image:: https://readthedocs.org/projects/geoai2025/badge/?version=latest
   :target: https://geoai2025.rtfd.io/en/latest/
   :alt: Documentation Status




.. |License| image:: https://img.shields.io/badge/license-GNU%20GPL%20v3%2B-blue.svg
   :target: https://github.com/vcalderon2009/GeoAI2025/blob/master/LICENSE.rst
   :alt: Project License






