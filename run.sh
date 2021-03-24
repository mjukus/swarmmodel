#!/bin/bash
# request resources:
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:00:00
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
cd Output
# and run your program, timing it. Usage: python path/to/run_from_terminal.py <axisN> <nRod> <partAxisSep> <Nt> <timestep> <**kwargs>.
time python ../Program/run_from_terminal.py 10 4 3E-6 10000 1E-5