#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=0

set mydir=/home/ejones/codes/thesis_code_snippets

rm $mydir/output/omp_struct_mapping.txt
rm $mydir/slurm/*.out
rm $mydir/*.out

exec $mydir/exec/omp_struct_mapping >> $mydir/output/omp_struct_mapping.txt
