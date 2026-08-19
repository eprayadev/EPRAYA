Hamiltonian's functions
=======================

All interactions present in the EPR hamiltonian are described with matricial functions. Here are their definitions and the auxiliary functions for their construction.

Interaction's functions
-----------------------

.. currentmodule:: base_ham

.. toctree::
   :caption: Interaction's functions
   :hidden:
   
   Hze
   Nhze
   Hfi
   Qii
   Iee
   Lorbit
   StevensO


.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Hze <Hze>`
     - Function for the Zeeman interaction.
   * - :doc:`epraya.Nhze <Nhze>`
     - Function for the nuclear Zeeman interaction.
   * - :doc:`epraya.Hfi <Hfi>`
     - Function for the hiperfine interaction.
   * - :doc:`epraya.Qii <Qii>`
     - Function for the nuclear quadrupolar interaction.   
   * - :doc:`epraya.Iee <Iee>`
     - Function for the electron-electron interaction.   
   * - :doc:`epraya.Lorbit <Lorbit>`
     - Function for the spin orbit interaction.       
   * - :doc:`epraya.StevensO <StevensO>`
     - Function for the expanded Stevens operators.     
     
     
     
Auxiliary functions
-------------------
.. currentmodule:: base_ham

.. toctree::
   :caption: Auxiliary functions
   :hidden:
   
   Kronecker
   Pauli
   chaframe
   Rotationmat
   Rotmatrix
   Convtarray
   Msmi
   Assingstatestobasis
   PMsmi
   MMsmi
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Kronecker <Kronecker>`
     - Function for the Kronecker delta.
   * - :doc:`epraya.Pauli <Pauli>`
     - Calculates the Pauli matrices for the spin system.
   * - :doc:`epraya.chaframe <chaframe>`
     - Rotates the hamiltonian parameters to the right reference frame using the Euler angles.    
   * - :doc:`epraya.Rotationmat <Rotationmat>`
     - Creates the rotation matrix for the Lab to sample frame transformation.    
   * - :doc:`epraya.Rotmatrix <Rotmatrix>`
     - Creates the rotation matrix using the Euler angles.       
   * - :doc:`epraya.Convtarray <Convtarray>`
     - Makes sure hamiltonian parameters have the right dimensions.      
   * - :doc:`epraya.Msmi <Msmi>`
     - Finds the quantum numbers of the operators and the possibles energy levels transitions.
   * - :doc:`epraya.Assingstatestobasis <Assingstatestobasis>`
     - Relates the eigenvectors of the hamiltonian in the basis of s and i with it's quantum numbers.   
   * - :doc:`epraya.PMsmi <PMsmi>`
     - Determinates the quantum numbers for the spin, nuclear spin and angular momentum.
   * - :doc:`epraya.MMsmi <MMsmi>`
     -  Determinates the quantum numbers of the operators and classify the transitions between energy levels for multisystems.
