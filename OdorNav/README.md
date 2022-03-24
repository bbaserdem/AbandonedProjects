# Odor Plume Navigation

A simulation of an odor plume, and an agent that learns to navigate it using RL.

# Requirements

The software is run on

* Matlab 2019a
* Python 3.7.3
* Numpy 1.16.2-1

# SPH

For simulating odor plume, an SPH solver is written in python.

* I need to calculate pairwise distances below a certain threshold.
* Needs to generalize easily to arbitrary dimensions. (2D to 3D)
* Needs to run fast when threshold is low and particle density is large.
* Needs to be able to do this on a torus. (Simplify boundary conditions)

This description of problem requires either wasted computational power,
or building of an algorithm.

For this to work, I have come up with the algorithm of the next section.
Implementation wise, the particles will be moving around.
The tradeoff with this algorithm is rapid-insertion data structures are needed,
which MATLAB does not have.
So this neccessitates the building of a specialized linked-list.
