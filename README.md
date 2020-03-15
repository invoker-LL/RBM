# Restricted Boltzmann Machine
[RBM](https://uwaterloo.ca/data-analytics/sites/ca.data-analytics/files/uploads/files/dbn2.pdf) in Pytorch

Based on [Hinton's MATLAB RBM script](www.sciencemag.org/cgi/content/full/313/5786/504/DC1).

This program trains Restricted Boltzmann Machine in which visible, binary, stochastic pixels are connected to hidden, stochastic real-valued feature detectors drawn from a unit
variance Gaussian whose mean is determined by the input from the logistic visible units. 

Learning is done with 1-step Contrastive Divergence.
