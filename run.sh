#!/bin/bash
# request resources:
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:00:00
# on compute node, change directory to 'submission directory':
cd Swarm/Output
# run your program, timing it. Usage: ./path/to/run_from_terminal.py <axisN> <nRod> <partAxisSep> <Nt> <timestep> <**kwargs>.
time ./Swarm/Program/run_from_terminal.py 10 4 3E-6 2000 1E-5