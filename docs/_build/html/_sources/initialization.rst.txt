
System initialization
=====================

Every simulation requires the initialization of the data containers ``Ham``, ``Exp`` and ``Vary``. This can be done using the functions ``Start(num)``, ``Jstart()`` or ``Jmstart()``, for classic, JAX and multisystem JAX, respectively, or by calling the classes ``Hval()``, ``Eco()`` and ``Eva()`` directly.

.. rubric:: Functions:

.. currentmodule:: base_ham

.. toctree::
   :hidden:
   
   Hval
   Eco
   Eva
   Multham
   Mulexco
   Muleva
   Start
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Hval <Hval>`
     - Container class for the hamiltonial's parameters.
   * - :doc:`epraya.Eco <Eco>`
     - Container class for the experimental parameters.
   * - :doc:`epraya.Eva <Eva>`
     - Container class for the variation of parameters.
   * - :doc:`epraya.Multham <Multham>`
     - Container class for the hamiltonial's parameters of ``num`` systems.
   * - :doc:`epraya.Mulexco <Mulexco>`
     - Container class for the experimental parameters of ``num`` systems.
   * - :doc:`epraya.Muleva <Muleva>`
     - Container class for the variation of parameters of ``num`` systems.
   * - :doc:`epraya.Start <Start>`
     - Function for Ham, Exp and Vary initialization for ``num`` systems.

JAX implementation
^^^^^^^^^^^^^^^^^^

.. rubric:: Functions:

.. currentmodule:: base_jax

.. toctree::
   :hidden:
   
   JHval
   JEco
   JEva
   Mjhval
   JEmco
   JEmva
   Jstart
   Jmstart
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JHval <JHval>`
     - JAX container class for the hamiltonial's parameters.
   * - :doc:`epraya.JEco <JEco>`
     - JAX container class for the experimental parameters.
   * - :doc:`epraya.JEva <JEva>`
     - JAX container class for the variation of parameters.
   * - :doc:`epraya.Mjhval <Mjhval>`
     - JAX container class for the hamiltonial's parameters of two systems.
   * - :doc:`epraya.JEmco <JEmco>`
     - JAX container class for the experimental parameters of two systems.
   * - :doc:`epraya.JEva <JEmva>`
     - JAX container class for the variation of parameters of two systems.
   * - :doc:`epraya.Jstart <Jstart>`
     - Function for Ham, Exp and Vary initialization for one system.
   * - :doc:`epraya.Jmstart <Jmstart>`
     - Function for Ham, Exp and Vary initialization for two systems.  
