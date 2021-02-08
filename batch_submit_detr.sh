#!/bin/bash
## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=detr
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/detr-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/detr-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task 10
#SBATCH --time=72:00:00
#SBATCH --mem=450G


### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
source /private/home/padentomasello/code/scripts/load_modules.sh

df -h
ls /scratch/slurm_tmpdir/
BUILD_DIR=/scratch/slurm_tmpdir/$SLURM_JOB_ID/$1/
ls $BUILD_DIR
mkdir -p $BUILD_DIR

mkdir -p $BUILD_DIR/lib/
cp /private/home/padentomasello/usr/lib/* $BUILD_DIR/lib/

cd $BUILD_DIR
git clone git@github.com:padentomasello/flashlight.git
cd flashlight && git fetch && git checkout $1
mkdir build && cd build
cmake .. -DArrayFire_DIR=/private/home/jacobkahn/usr/share/ArrayFire/cmake -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA -DFL_BUILD_VISION=ON -DFL_LIBRARIES_USE_KENLM=OFF -DFL_BUILD_TESTS=OFF
make -j $(nproc) Detr
cd ..

### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each node. However I am using
### the wrapper.sh example I showed before so that each task only
### sees one GPU
mpirun wrapper_detr.sh $1
