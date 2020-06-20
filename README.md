# A bio-inspired bistable recurrent cell allows for long-lasting memory
A repository containing the code for the "A bio-inspired bistable recurrent cell allows for long-lasting memory" paper.

Link to the paper : https://arxiv.org/abs/2006.05252

## Dependencies
The only dependency for the code provided is tensorflow 2 (and numpy, which is installed by default with tensorflow).

Install tensorflow : https://www.tensorflow.org/install

## Run experiments
Each benchmark included in the paper is associated to a script. These scripts simply use a Sequential model of Keras and 
integrate the corresponding recurrent cells. All of the three scripts are extremely similar but were built separately for 
simplicity and self-containedness.
* The copy first input benchmark can be launched through the "benchmark1.py" script.
* The denoising benchmark can be launched through the "benchmark2.py" script.
* The sequential MNIST benchmark can be launched through the "benchmark3.py" script.


