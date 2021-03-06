# Machine Learning for Source Location
More details to follow. For the moment there are just two component codes.
 
## `blockdata.py`

A digital twin for a lab rig.

*To solve a 3D linear quasistatic or viscodynamic problem using python and FEniCS <http://fenicsproject.org>. The right hand side forcing is narrowly located at a randomly chosen point in the domain, vibrating at a carrier frequency which is then modulated by a lower frequency. The response is then picked up by microphones and accelerometers at given points on the boundary.*

The bar has dimensions 30cm long, by 10 wide and 5 deep. The bottom is considered to be fixed in position for all time (homogeneous essential - Dirichlet - boundary condition) and the remaining faces are traction free (homogeneous natural - Neumann - boundary condition).

Based on `demo_elastodynamics.py` as downloaded on 10 Dec 2019 from
<https://fenicsproject.org/docs/dolfin/latest/python/demos/elastodynamics>
and also on `ft06_elasticity.py` as downloaded on 10 Dec 2019 from
<https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft06_elasticity.py>

The code is awaiting many mods:

- addition of a quasistatic viscoelastic capability using internal variables
- addition of a quasistatic viscoelastic capability using fractional calculus 'power laws'
- addition of at least one form of time stepper for the dynamic problem(s)

Eventually this will be used to generate virtual training data for a machine learning source identification problem, as well as the test and validation data. We would like more than one forward solver in order to address so-called *inverse crime*

## `postpro.py`

*To post process sensor data from `blockdata.py` and compress it to a form suited to training stage the Neural Net.*

The code is awaiting many mods:

- intelligent and variable interfacing to the `blockdata.py` code so that post-processing is seamless. 


## `ihcc_linear_elastostatic_exact_data.py`
Contains FEniCS expressions for generating manufactured exact solutions for checking consistency/convergence.