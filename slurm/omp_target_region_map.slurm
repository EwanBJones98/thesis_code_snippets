#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=0

set mydir=/home/ejones/codes/thesis_code_snippets

rm $mydir/output/omp_target_region_map.txt
rm $mydir/slurm/*.out
rm $mydir/*.out

exec $mydir/exec/omp_target_region_map >> $mydir/output/omp_target_region_map.txt