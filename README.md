Truncated PCA Benchmark
===============================

Problem definition
------------------

TODO: Complete

Install
--------

This benchmark can be installed using the following commands:

```bash
   $ pip install git+https://github.com/benchopt/benchopt.git@d68718d
   $ git clone https://github.com/MortimerTP/benchmark_pca
   $ benchopt install benchmark_pca
```

To run the benchmark, options can be passed to ``benchopt run`` to restrict the benchmarks to some solvers or datasets, e.g.:

```bash
   $ benchopt run benchmark_pca -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10
````

You can also use config files to set the benchmark run:

```bash
   $ benchopt run benchmark_pca --config config/X.yml
```

where ``X.yml`` is a config file. See https://benchopt.github.io/index.html#run-a-benchmark for an example of a config file. This will launch a huge grid search. When available, you can rather use the file ``X_best_params.yml`` to launch an experiment with a single set of parameters for each solver.

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

### If you want to add a solver or a new problem, you are welcome to open an issue or submit a pull request!  



