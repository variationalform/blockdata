# `blockdata.py`

*To solve a 3D linear quasistatic or viscodynamic problem using python and FEniCS <http://fenicsproject.org>. The right hand side forcing is narrowly located at a randomly chosen point in the domain, vibrating at a carrier frequency which is then modulated by a lower frequency. The response is then picked up my microphones and acceleromenters at given points on the boundary.*

The code is awaiting many mods:

- addition of a quasistatic viscoelastic capability using internal variables
- addition of a quasistatic viscoelastic capability using fractional calculus 'power laws'
- addition of at least one form of time stepper for the dynamic problem(s)

Eventually this will be used to generate virtual training data for a machine learning source identification problem, as well as the test and validation data. We would like more than one forward solver in order to address so-called *inverse crime*
