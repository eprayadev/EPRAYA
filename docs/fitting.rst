Fitting
=======
Here can be find the functions to load and fit experimental data. EPRAYA has 5 methods for EPR data fitting: Nelder-Mead, Genetic Algorithm, Metrópolis, Least Squares and the use of the ADAM Algorithm. 

Data load functions
-------------------

EPRAYA has different forms of loading data, based in the *np.load* function. The choosing between this functions depends on the user and the possibility of use the python package *tkinter*.

.. currentmodule:: base_plot

.. toctree::
   :caption: Data load functions
   :hidden:

   Sload
   Splot
   Sfilter
   Overseer
   Sload1
   Spmanipulation
   Seek
   Termal

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Sload <Sload>`
     - Basic function for data load of counts and fields.
   * - :doc:`epraya.Splot <Splot>`
     - Plotting function for field and count data.    
   * - :doc:`epraya.Sfilter <Sfilter>`
     - Applys a Savitzky-Golay filter to the count data, finds the resonant fields and peak to peak width, and generates the plots of the filtered data, its first and second integrals.
   * - :doc:`epraya.Overseer <Overseer>`
     - Interactive function for spectrum analysis of the experimental data, applying Savitzky-Golay filter to the data, finding it's resonant fields, peak to peak distance and first and second integral.
   * - :doc:`epraya.Sload1 <Sload1>`
     - Basic function for data load of counts and fields in Seek.     
   * - :doc:`epraya.Spmanipulation <Spmanipulation>`
     -  Applys a Savitzky-Golay filter to the count data, finds the resonant fields and peak to peak width, and generates the plots of the filtered data, its first and second integrals, using the values of the sliders.
   * - :doc:`epraya.Seek <Seek>`
     -  Function for data loading and initial analysis, based on tkinter.
   * - :doc:`epraya.Termal <Termal>`
     -  Function to analyze the change in peak to peak distance, resonant field position, first and second integral of EPR spectrum data with the temperature.
     
     
Fitting functions
-----------------

This methods can be use directly or with the wrap function *Fitting*, however, the ADAM method requieres the use of the *JAX* framework of EPRAYA, described in :doc: `jax_impl`.  

.. currentmodule:: base_fit

.. toctree::
   :caption: Fitting functions
   :hidden:
   
   Fitting
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Fitting <Fitting>`
     - Wrap function for the data fitting process.  
   
