Jax implementation
=======================

Here are the JAX function for the simualtion and fitting of EPR cw data. 

All functions follow a similar approach to that of its counterparts for the *Classic* method described in the other sections, but adapted to make use of the JAX advantages, without requiring a large amount of RAM.

Hamiltonian's functions
-----------------------

All interactions present in the EPR hamiltonian are described with jax.numpy matricial functions. Here are their definitions and the auxiliary functions for their construction.

Interaction's functions
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Interaction's functions
   :hidden:
   
   JHze
   JNhze
   JHfi
   JQii
   JIee
   JLorbit
   JStevensO
  
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JHze <JHze>`
     - Function for the Zeeman interaction.
   * - :doc:`epraya.JNhze <JNhze>`
     - Function for the nuclear Zeeman interaction.
   * - :doc:`epraya.JHfi <JHfi>`
     - Function for the hiperfine interaction.
   * - :doc:`epraya.JQii <JQii>`
     - Function for the nuclear quadrupolar interaction.   
   * - :doc:`epraya.JIee <JIee>`
     - Function for the electron-electron interaction.   
   * - :doc:`epraya.JLorbit <JLorbit>`
     - Function for the spin orbit interaction.       
   * - :doc:`epraya.JStevensO <JStevensO>`
     - Function for the expanded Stevens operators.     
     
Auxiliary functions
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Auxiliary functions
   :hidden:
   
   JKronecker
   JPauli
   Jchaframe
   JRotationmat
   JRotmatrix
   JConvtarray
   JMsmi
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JKronecker <JKronecker>`
     - Function for the Kronecker delta.
   * - :doc:`epraya.JPauli <JPauli>`
     - Calculates the Pauli matrices for the spin system.
   * - :doc:`epraya.Jchaframe <Jchaframe>`
     - Rotates the hamiltonian parameters to the right reference frame using the Euler angles.    
   * - :doc:`epraya.JRotationmat <JRotationmat>`
     - Creates the rotation matrix for the Lab to sample frame transformation.    
   * - :doc:`epraya.JRotmatrix <JRotmatrix>`
     - Creates the rotation matrix using the Euler angles.       
   * - :doc:`epraya.JConvtarray <JConvtarray>`
     - Makes sure hamiltonian parameters have the right dimensions.      
   * - :doc:`epraya.JMsmi <JMsmi>`
     - Determinates the quantum numbers for the spin, nuclear spin and angular momentum.
