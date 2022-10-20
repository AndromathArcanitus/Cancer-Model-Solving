#!/bin/bash
#BSUB -J 1.5-1-1-1-1-1
#BSUB -o my_job_op31.txt
#BSUB -W 200:00
#BSUB -u yuhu@umass.edu
#BSUB -N
#BSUB -n 1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=1000]

singularity exec /home/yh30a/fenics-2019.2.0.dev0_w_gmsh_3.0.6.simg python3 ./Main4.py
