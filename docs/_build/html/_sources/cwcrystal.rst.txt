Monocristal samples
===================

Includes the functions for monocristal samples simulation (one fixed orientation), with the implementation of the Stoll *Easyspin* method for the resonant fields determination. Also includes the energy diagrams generator function.

Many of the functions have the same names that the powder samples' ones, however their implementation differs, because only one orientation is calculated.

Intensity and resonant fields functions
---------------------------------------
     
.. currentmodule:: base_powd

.. toctree::
   :caption: Intensity and resonant fields functions
   :hidden:
   
   Eresonant
   Plotsim
   Cristalfm
   Music
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Eresonant <Eresonant>`
     - Function for the simulation of the EPR spectrum for monocrystal samples.
   * - :doc:`epraya.Plotsim <Plotsim>`
     - Function to produce the graph of the spectrum and the energy diagram for monocrystal samples.  
   * - :doc:`epraya.Cristalfm <Cristalfm>`
     - Function for the calculation of the spectrum of monocristals without producing the graphs. 
   * - :doc:`epraya.Music <Music>`
     - Wrap function that calculates the spectrum and table of transitions of multisystems.     
     
Eigen values and vectors functions
----------------------------------

.. currentmodule:: base_powd

.. toctree::
   :caption: Eigen values and vectors functions
   :hidden:
   
   EAdaptarray
   EHungorder
   ERetrack
   Mulgetlabel

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.EAdaptarray <EAdaptarray>`
     - Finds the energy values and eigenvectors of the total hamiltonian.
   * - :doc:`epraya.EHungorder <EHungorder>`
     - Solves the assigment problem with the J-V method implemented in Scipy for the eigenvectors and energy values.    
   * - :doc:`epraya.ERetrack <ERetrack>`
     - Relates the energy values and eigenvectors to the quantum numbers of the system.
   * - :doc:`epraya.Mulgetlabel <Mulgetlabel>`
     - Creates the state ket for the energy level, using the quantum numbers.
     

