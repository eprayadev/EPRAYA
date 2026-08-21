Fitting
=======
Here can be find the functions to load and fit experimental data. EPRAYA has 5 methods for EPR data fitting: Nelder-Mead, Genetic Algorithm, Metrópolis, Least Squares and the use of the ADAM Algorithm. This methods can be use directly or with the wrap function *Fitting*, however, the ADAM method requieres the use of the *JAX* framework of EPRAYA, described in :doc: `jax_impl`.  

Data load functions
-------------------

EPRAYA has different forms of loading data, based in the *np.load* function. The choosing between this functions depends on the user and the possibility of use the python package *tkinter*.

.. currentmodule:: base_plot

.. toctree::
   :caption: Data load functions
   :hidden:
   
   Eresonant

   
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
     

