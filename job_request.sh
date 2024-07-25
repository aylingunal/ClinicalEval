#!/bin/bash
#PBS -lselect=4:ncpus=3:mpiprocs=3:mem=4gb
#PBS -l place=scatter
#PBS -lwalltime=0:10:00
#PBS -m abe
#PBS -M gunala@umich.edu
#PBS -o /home/agunal/scratch/goldkind-clinical-ai/tmpdir/
#PBS -e /home/agunal/scratch/goldkind-clinical-ai/tmpdir/

TMPDIR=/home/agunal/scratch/goldkind-clinical-ai/tmpdir/job_outputs/
export TMPDIR

# module load ucx
module load anaconda/default
module load python311
module load openmpi4/gcc/4.1.5

# activate my local env
source /home/agunal/miniconda3/etc/profile.d/conda.sh
conda activate ClinicalEvalEnv
# execute my script
cd /home/agunal/ClinicalEval/pl/
mpiexec --mca btl '^openib' /home/agunal/miniconda3/envs/ClinicalEvalEnv/bin/python dpo.py
# deactivate
conda deactivate