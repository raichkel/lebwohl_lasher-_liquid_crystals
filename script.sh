#!/bin/bash

#SBATCH --job-name=LL_crystal
#SBATCH --partition=teach_cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=phys026162

## Direct output to the following files.
## (The %j is replaced by the job id.)
#SBATCH -e _%j.txt
#SBATCH -o out_%j.txt

# Just in case this is not loaded already...
module load languages/intel/2020-u4
module add languages/anaconda3/2022.11-3.9.13

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"

# Submit
mpiexec -n 4 python LebwohlLasher.py 3 20 1.0 1

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"