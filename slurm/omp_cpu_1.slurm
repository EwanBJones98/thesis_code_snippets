#!/bin/tcsh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set mydir=/home/ejones/codes/thesis_code_snippets

rm $mydir/output/omp_cpu_1.txt
rm $mydir/slurm/*.out
rm $mydir/*.out

exec $mydir/exec/omp_cpu_1 >> $mydir/output/omp_cpu_1.txt