# A bio-inspired bistable recurrent cell allows for long-lasting memory
A repository containing the code for the "A bio-inspired bistable recurrent cell allows for long-lasting memory" paper.

Link to the paper : https://arxiv.org/abs/2006.05252

## Dependencies
The only dependency for the code provided is tensorflow 2 (and numpy, which is installed by default with tensorflow).

Install tensorflow : https://www.tensorflow.org/install

## Minimal working example
We advise to first look at the "MinimalWorkingExample.py" script, as it is self-contained and can be
used to reproduce the results on the "copy first input" benchmark. It can also easily be modified to reproduce the results 
for both other benchmarks.

## Run experiments
Each benchmark included in the paper is associated to a script. As opposed to the MinimalWorkingExample,
these scripts use networks which are not built directly through the Sequential model of Keras. Rather, we use a 
"CustomModel" class, which we built for modularity. Note however, that the "MinimalWorkingExample" script provides 
all that is required to use BRC and nBRC cells for all the benchmarks.
* The copy first input benchmark can be launched through the "benchmark1.py" script.
* The denoising benchmark can be launched through the "benchmark2.py" script.
* The sequential MNIST benchmark can be launched through the "benchmark3.py" script.


