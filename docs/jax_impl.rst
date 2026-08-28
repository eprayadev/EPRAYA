Jax implementation
==================

Here are the *JAX* function for the simualtion and fitting of EPR cw data. 

All functions follow a similar approach to that of its counterparts for the *Classic* method described in the other sections, but adapted to make use of the *JAX* advantages, without requiring a large amount of RAM.

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

Powder samples
--------------

Like its classic counterpart, the process is based in the modificated SOPHE method (using the function *epraya.Delaunay*), however, the calculation is divided in blocks to keep the processing speed. 

*Note: By the nature of the JAX framework, the reassigment of values to variables and conditionals evaluations must be performed using JAX functions, which requires modifying the program logic, but the results are equivalent to those of the classic one.*

Intensity and resonant fields functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Intensity and resonant fields functions
   :hidden:
   
   
   JPowder
   JCalpowder
   JMulpol
   Jcalmulta
   oneori
   JNresina
   JBoltfactor
   JCaltriangle
   Meshtriangle

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JPowder <JPowder>`
     - Wrap function for the simulation of the EPR spectrum for powder samples.
   * - :doc:`epraya.JCalpowder <JCalpowder>`
     - Function for the simulation of the EPR spectrum for powder samples using the JAX functions. 
   * - :doc:`epraya.JMulpol <JMulpol>`
     - Wrap function for the simulation of the EPR spectrum for powder samples of two systems.
   * - :doc:`epraya.Jcalmulta <Jcalmulta>`
     - Determinates the spectrum for a two paramagnetic centers system.
   * - :doc:`epraya.oneori <oneori>`
     - Wrap function to use JPadaptarray and JNresina with the JAX.vmap implementation. 
   * - :doc:`epraya.JNresina <JNresina>`
     - Determinates the resonant fields and intensities of the spectrum using the expression for the first order perturbation limit. 
   * - :doc:`epraya.JBoltfactor <JBoltfactor>`
     - Calculates the Boltzmann distribution for the intensity, using states *di* and *dj* and their related energies.
   * - :doc:`epraya.JCaltriangle <JCaltriangle>`
     - Creates a sketch spectrum using a barycentral mesh of triangles to calculate the contribution of the resonant fields and their intensities in the final spectrum.
   * - :doc:`epraya.Meshtriangle <Meshtriangle>`
     - Creates the baricentral mesh of triangles for the JCaltriangle function.


Line profile functions
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Line profile functions
   :hidden:
   
   JLorentzp
   JGaussp
   JVoigtp

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JLorentzp <JLorentzp>`
     - Determinates the lorentzian profile of the spectrum.
   * - :doc:`epraya.JGaussp <JGaussp>`
     - Determinates the gaussian profile of the spectrum.
   * - :doc:`epraya.JVoigtp <JVoigtp>`
     - Determinates the voigtian profile of the spectrum.
     
Eigen values and vectors functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: base_jax

.. toctree::
   :caption: Eigen values and vectors functions
   :hidden:
   
   JPadaptarray
   Hungarian
   Jungarian
   JPretrack	
   containeigh
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.JPadaptarray <JPadaptarray>`
     - Constructs the Zeeman hamiltonian, adds it to the complete one and finds the energy values and eigenvectors.
   * - :doc:`epraya.Hungarian <Hungarian>`
     - Solves the assigment problem with the J-V method implemented in Scipy for the eigenvectors and energy values.    
   * - :doc:`epraya.Jungarian <Jungarian>`
     - Wrap function to call the Scipy J-V method outside of JAX using ShapeDtypeStruct and pure_callback.
   * - :doc:`epraya.JPretrack <JPretrack>`
     - Organize the eigenvectors and energies to relate them with the quantum numbers of the system.
   * - :doc:`epraya.containeigh <containeigh>`
     - Wrap function for the eignevalues determination using JAX.	
     
Monocristal samples
-------------------

Like its classic counterpart, the process includes the functions for monocristal samples simulation (one fixed orientation), with the implementation of the Stoll *Easyspin* method for the resonant fields determination. Also includes the energy diagrams generator function.

Intensity and resonant fields functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     
.. currentmodule:: base_jax

.. toctree::
   :caption: Intensity and resonant fields functions
   :hidden:
   
   Jresonant
   Calresonant
   JMusic
   Jcalmusic
   
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :doc:`epraya.Jresonant <Jresonant>`
     - Wrap function for the simulation of the EPR spectrum for monocrystal samples.
   * - :doc:`epraya.Calresonant <Calresonant>`
     - Function for the calculation of the EPR cw spectrum of monocristal systems.
   * - :doc:`epraya.JMusic <JMusic>`
     - Wrap function for the simulation of the EPR spectrum for monocristal samples of two systems.
   * - :doc:`epraya.Jcalmusic <Jcalmusic>`
     - Determinates the spectrum for a two paramagnetic centers monocristal system..     
