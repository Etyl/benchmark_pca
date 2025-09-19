Various remarks and "would have been nice to implement", ranked by priority in each section
=======

Datasets 
===

bigger datasets sources - https://sdrbench.github.io/ (scientific data compression benchmark) ; https://gdex.ucar.edu/ (previous host of ISABEL dataset, lot of climate simulations) ; probably a lot of others 

data loading - (in solvers) currently only sampling with replacement, no true mini-batch with replacement like in DL

Compute a pseudo-GT for each dataset (may be expensive, RBKI with dask if too big)

Solvers 
===

Sparser projection schemes (using Cholesky decomposition or projecting every few iterations)

Deflation ([aka Lazy SVD](https://arxiv.org/pdf/1607.03463))

Basic subspace iteration (/ Power method) 

Add an incremental PCA/SVD (not directly the one from scikit learn as it also perform online mean substraction, but same family of algo)

Other optimization methods (line-search ; admm ; variance reduced stochastic riemannian?)

Objective
===

Add GT dependent metrics (subspace distance, angular streaks, angular distances ...)

Add other compression metrics from SDRBench


Various
===

Investigate subspace tracking techniques and use it to quantify (not too expensively) how much the PCA basis may change along a simulation  

Better understanding of LOBPCG (<3) and how to exploit our current estimate to further accelerate convergence 

Benchopt 
===

Hyperparameter optimization 

When an algorithm doesn't always respect the constraints, de-count projection from evaluation time ? (valable also for the question of the distribed evaluation of the mean parameters vs each single node parameters) 

Bugs : solver has no attributes .components for cache of a run_once algorithm ; 