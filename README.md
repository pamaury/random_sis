# Quantum algorithm for discrete Gaussian sampling

This is the code to compute the estimates in the paper.

## Reproducing the results

To reproduce the results, we strongly suggest to use Docker.
We provide a Dockerfile to build a docker image that contains
everything you need. Assuming you have docker installed on your machine,
you can run the following command to build the docker image:

```console
docker build -t random_sis .
```

You can then run the container as follows:

```console
# Make sure to run it from the directory containing estimator_sis.py
docker run -v $(pwd):/home/sage/mnt -it random_sis
```

To reproduce the numerical experiments of the dual attack, run the following
commands *inside the container*:

```console
# run in the container:
sage
```
then enter the following commands *inside sage*
```py
import sys
sys.path.append('/home/sage/lattice-estimator')
attach("estimator_sis.py")
# Option 1: run the optimizer.
%time results = runall()
# optional: produce the latex tables of the article
print(results_table_latex(results, quantum=True))
print(results_table_latex(results, quantum=True, red_cost_model=BCSS23()))
print(results_table_latex(results, quantum=True, red_cost_model=QuantumABFKSW20))
# Option 2: just reproduce the results of the paper
reproduce_paper()
print(reproduce_paper(QUANTUM_SIS_PARAMS_BCSS23))
print(reproduce_paper(QUANTUM_SIS_PARAMS_CLASSICAL))
print(reproduce_paper(QUANTUM_SIS_PARAMS_QABFKSW20))
```

**Technical details:** the docker image built does not include the code, instead
it expects the code to be available at the mount point /home/sage/mnt inside
the container. This allows you to edit the code outside of the container
while running the container.
