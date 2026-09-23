Fitting
=======
Here can be find the functions to load and fit experimental data. EPRAYA has 5 methods for EPR data fitting: Nelder-Mead, Genetic Algorithm, Metrópolis, Least Squares and the use of the ADAM Algorithm. The results of the fitting are print in the screen and are also contain in epr.restult.

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

This methods can be use directly or with the wrap function *Fitting*, however, the *Briggs* function for the ADAM method requieres the use of the *JAX* framework of EPRAYA, described in :doc:`jax_impl`.  

The variance calculated for the fitted parameters is statistical and avoids the calculation for parameters that have a high jacobian correlation number. 

.. note:: 
   
   The resolution of the grid for powder samples (M=70) converges for the majority of EPR solid cw spectrum, making it related discretation error (DE) negligible for the calculation of the variance. However, it is recommended to change M to a higher resolution (M=90 or M=120) in the following cases:
   
   
   * High anisotropies or low Hpp: Where the gradients of tensores g and A require higher resolutions.
   * High frequency spectrums.
   * Verification of convergence: To prove the fit validity.
   


.. currentmodule:: base_fit

.. toctree::
   :caption: Fitting functions
   :hidden:
   
   
   Fitting
   Fitresult
   Selectvarian
   Nelder
   Nelder1
   Nelder2
   Genio
   Genio1
   Genio2
   Metro
   Metro1
   Metro2
   LSquare
   LSquare1
   LSquare2
   Briggs
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Fitting <Fitting>`
     - Wrap function for the data fitting process.  
   * - :doc:`epraya.Fitresult <Fitresult>`
     - Container for the results of the Fitting function.
   * - :doc:`epraya.Selectvarian <Selectvarian>`
     - Calculates the variance of the fit using the Jacobian matrix of the residuals.  
   * - :doc:`epraya.Nelder <Nelder>`
     - Implementation of the Nelder Mead algorithm for fitting data.   
   * - :doc:`epraya.Nelder1 <Nelder1>`
     - Implementation of the Nelder Mead algorithm for fitting data for a single system.
   * - :doc:`epraya.Nelder2 <Nelder2>`
     - Implementation of the Nelder Mead algorithm for fitting data for multisystems.
   * - :doc:`epraya.Genio <Genio>`
     - Fitting function for the experimental data using the genetic algorithm.
   * - :doc:`epraya.Genio1 <Genio1>`
     - Fitting function for the experimental data using the genetic algorithm for a single system.
   * - :doc:`epraya.Genio2 <Genio2>`
     - Fitting function for the experimental data using the genetic algorithm for multysystems.
   * - :doc:`epraya.Metro <Metro>`
     - Fitting adjutsment of the experimental data using a modified Metrópolis-Simulated annealing approach.
   * - :doc:`epraya.Metro1 <Metro1>`
     - Fitting adjutsment of the experimental data using a modified Metrópolis-Simulated annealing approach for a single system.
   * - :doc:`epraya.Metro2 <Metro2>`
     - Fitting adjutsment of the experimental data using a modified Metrópolis-Simulated annealing approach for multiple systems.
   * - :doc:`epraya.LSquare <LSquare>`
     - Fitting adjutsment of the experimental data using the *scipy.optimize.least_squares* method.
   * - :doc:`epraya.LSquare1 <LSquare1>`
     - Fitting adjutsment of the experimental data using the *scipy.optimize.least_squares* method. This case is for simple systems.
   * - :doc:`epraya.LSquare2 <LSquare2>`
     - Fitting adjutsment of the experimental data using the *scipy.optimize.least_squares* method. This case is for multiple systems.
   * - :doc:`epraya.Briggs <Briggs>`
     - Wrap function for the ADAM fitting method.     

