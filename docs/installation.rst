Installation
============

The recommended method of installing EPRAYA is environment-based using the ``pip`` command.

Installing with pip
-------------------

If you are using a notebook or distribution, the installation can be done by executing the following command on the notebook's cells or terminal, respectively:

.. code-block:: bash

   pip install epraya

In other cases, use the terminal and follow this step-by-step guide:

1. Install `Python <https://www.python.org/downloads/>`_.

2. Create and activate a virtual environment with ``venv``. In this example, we use **myenv**:

   **Ubuntu and Debian**

   .. code-block:: bash

      sudo apt install python3-venv
      python3 -m venv myenv
      source myenv/bin/activate

   **Fedora**

   .. code-block:: bash

      sudo dnf install python3-venv
      python3 -m venv myenv
      source myenv/bin/activate

   **Mac**

   .. code-block:: bash

      python3 -m venv myenv
      source myenv/bin/activate

   **Windows Cmd**

   .. code-block:: bash

      python -m venv myenv
      .\myenv\Scripts\activate

3. Install EPRAYA using ``pip``:

   .. code-block:: bash

      pip install epraya

4. Make sure ``tkinter`` is installed (by default it's in the Python download, but it can create problems later if missing).

   **Ubuntu and Debian**

   .. code-block:: bash

      sudo apt install python3-tk

   **Fedora**

   .. code-block:: bash

      sudo dnf install python3-tk

Now you are ready to simulate EPR data with EPRAYA!
