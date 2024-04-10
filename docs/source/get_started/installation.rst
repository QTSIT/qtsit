Installation
============

This page is a guide on how to install qisit

Stable version
--------------

Install qtsit via pip or conda by simply running,

.. code-block:: bash

    pip install qtsit

or 

.. code-block:: bash

    conda install -c conda-forge qtsit

Nightly build version
---------------------
The nightly version is built by the HEAD of qtsit.


.. code-block:: bash

    pip install --pre qtsit

From source with conda
----------------------

**Installing via these steps will ensure you are installing from the source**.

**Prerequisite**

- Shell: Bash, Zsh, PowerShell
- Conda: >4.6


First, please clone the qtsit repository from GitHub.

.. code-block:: bash

    git clone https://github.com/QTSIT/qtsit.git
    cd qtsit


Then, execute the shell script. The shell scripts require two arguments,
**python version** and **gpu/cpu**.

.. code-block:: bash

    source scripts/install_qtsit_conda.sh 3.10 cpu

If you are using the Windows and the PowerShell:

.. code-block:: ps1

    .\scripts\install_qtsit_conda.ps1 3.10 cpu

| Sometimes, PowerShell scripts can't be executed due to problems in Execution Policies.
| In that case, you can either change the Execution policies or use the bypass argument.


.. code-block:: ps1

    powershell -executionpolicy bypass -File .\scripts\install_qtsit_conda.ps1 3.7 cpu

| Before activating qtsit environment, make sure conda has been initialized.
| Check if there is a :code:`(XXXX)` in your command line. 
| If not, use :code:`conda init <YOUR_SHELL_NAME>` to activate it, then:

.. code-block:: bash

    conda activate qtsit
    pip install -e .
    pytest -m  qtsit 


.. qtsit has soft requirements, which can be installed on the fly during development inside the environment 
.. but if you want to install all the soft-dependencies at once, then take a look at 
.. `qtsit/requirements <https://github.com/qtsit/qtsit/tree/main/requirements>`_
