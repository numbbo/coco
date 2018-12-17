#!/bin/sh
#$ -j y
#$ -pe smp 1
#$ -l h_rt=8:00:00
#$ -l h_vmem=2G
#$ -M v.volz@qmul.ac.uk
#$ -m bea

module load cuda/8.0.44
module load java/1.8.0_152-oracle
module load python/2.7.15
module load eigen/3.3.4
module load cmake/3.8.0
module load boost/1.63.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/home/eex642/gbea/code-experiments/rw-problems/top-trumps/TopTrumps

cd ../../build/c/
#$ -cwd

./example_experiment 2>&1 > output.ex.dat

