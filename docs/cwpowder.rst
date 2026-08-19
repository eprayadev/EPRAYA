Powder samples
==============

Includes the functions for powder samples simulation, with the implementation of a modified version of the *XSOPHE* method. 


Intensity and resonant fields functions
---------------------------------------
     
.. currentmodule:: base_powd

.. toctree::
   :caption: Intensity and resonant fields functions
   :hidden:
   
   Delaunay
   Powder
   Calpowder
   Mulpol
   Omegaparal
   Betaparal
   Nresina
   Boltfactor
   Caltriangle

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Delaunay <Delaunay>`
     - Implementation of a modificated version of the SOPHE method to make the grid for the powder system.
   * - :doc:`epraya.Powder <Powder>`
     - Wrap function for the simulation of the EPR spectrum for powder samples.  
   * - :doc:`epraya.Calpowder <Calpowder>`
     - Wrap function for the simulation of the EPR spectrum for powder samples.     
   * - :doc:`epraya.Mulpol <Mulpol>`
     - Function for the simulation the EPR spectrum for powder samples of multiple interactive or non interactive systems.      
   * - :doc:`epraya.Omegaparal <Omegaparal>`
     - Wrap function for the parallel calculation of the resonant fields and intensities of the spectrum.
   * - :doc:`epraya.Betaparal <Betaparal>`
     - Wrap function for the parallel calculation of the resonant fields and intensities of the spectrum for multisystems.
   * - :doc:`epraya.Nresina <Nresina>`
     - Determinates the resonant fields and intensities of the spectrum using the expression for the first order perturbation limit.
   * - :doc:`epraya.Boltfactor <Boltfactor>`
     - Calculates the Boltzmann distribution for the intensity.
   * - :doc:`epraya.Caltriangle <Caltriangle>`
     - Calculates the contribution of the resonant fields and their intensities in the final spectrum using a barycentral mesh of triangles.


Line profile functions
----------------------

.. currentmodule:: base_powd

.. toctree::
   :caption: Line profile functions
   :hidden:
   
   Lorentzp
   Gaussp
   Voigtp

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Lorentzp <Lorentzp>`
     - Determinates the lorentzian profile of the spectrum.
   * - :doc:`epraya.Gaussp <Gaussp>`
     - Determinates the gaussian profile of the spectrum.
   * - :doc:`epraya.Voigtp <Voigtp>`
     - Determinates the voigtian profile of the spectrum.
    
Eigen values and vectors functions
----------------------------------

.. currentmodule:: base_powd

.. toctree::
   :caption: Eigen values and vectors functions
   :hidden:
   
   Padaptarray
   Hungorder
   Pretrack
   Getlabel

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Padaptarray <Padaptarray>`
     - Finds the energy values and eigenvectors of the total hamiltonian.
   * - :doc:`epraya.Hungorder <Hungorder>`
     - Solves the assigment problem with the J-V method implemented in Scipy for the eigenvectors and energy values.    
   * - :doc:`epraya.Pretrack <Pretrack>`
     - Relates the energy values and eigenvectors to the quantum numbers of the system.
   * - :doc:`epraya.Getlabel <Getlabel>`
     - Creates the state ket for the energy level, using the quantum numbers.
     

