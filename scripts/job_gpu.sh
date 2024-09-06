#!/bin/sh
#PBS -lselect=1:ncpus=32:ngpus=4:mem=275gb
#PBS -m abe
#PBS -l walltime=48:00:00
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
conda activate tmpEnv
# execute my script
cd /home/agunal/scratch/goldkind-clinical-ai/dev/ClinicalEval/pl/
mpiexec --mca btl '^openib' /home/agunal/scratch/goldkind-clinical-ai/dev/miniconda3/envs/tmpEnv/bin/python dpo_eval.py
# deactivate
conda deactivate
