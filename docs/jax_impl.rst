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
   
   Hze
   Nhze
   Hfi
   Qii
   Iee
   Lorbit
   StevensO
   
Auxiliary functions
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Auxiliary functions
   :hidden:
   
   JKronecker
   JPauli
   chaframe
   Rotationmat
   Rotmatrix
   Convtarray
   Msmi
   Assingstatestobasis
   PMsmi
   MMsmi
