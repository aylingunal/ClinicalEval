#!/bin/bash
#PBS -l select=1:mem=12:ncpus=12
#PBS -l place=scatter
#PBS -l walltime=12:00:00
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
source /home/agunal/scratch/goldkind-clinical-ai/dev/miniconda3/etc/profile.d/conda.sh
conda activate ClinicalEvalEnv
# execute my script
cd /home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/iaa/
mpiexec --mca btl '^openib' /home/agunal/scratch/goldkind-clinical-ai/dev/miniconda3/envs/ClinicalEvalEnv/bin/python annot_embeds.py
# deactivate
conda deactivate