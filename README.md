# Bilevel as MPCC Experiments

This repository contains the code and data for the experiments in the paper "Bilevel as Mathematical Program with Equilibrium Constraints.."

## Installation
We need to build the docker image to run the experiments. To do so, run the following command:
```bash
$ docker build -t bilevel-mpcc .
```

## Running the experiments
To run the experiments, you can use the following command:
```bash
$ docker run -v $(pwd):/usr/src/app bilevel-mpcc python run_experiments.py
```
