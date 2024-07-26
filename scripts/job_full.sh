#!/bin/bash
#PBS -l select=4:mem=96:ncpus=24
#PBS -l place=scatter
#PBS -l walltime=336:00:00
#PBS -m abe
#PBS -M gunala@umich.edu
#PBS -o /home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs/
#PBS -e /home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs/

TMPDIR=/home/agunal/scratch/goldkind-clinical-ai/tmpdir/program_outputs/
export TMPDIR

# module load ucx
module load anaconda/default
module load python311
module load openmpi4/gcc/4.1.5

# activate env
source /home/agunal/miniconda3/etc/profile.d/conda.sh
conda activate ClinicalEvalEnv
# execute my script
cd /home/agunal/ClinicalEval/pl/
mpiexec --mca btl '^openib' /home/agunal/miniconda3/envs/ClinicalEvalEnv/bin/python dpo.py
# deactivate
conda deactivate