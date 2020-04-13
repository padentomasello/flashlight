#!/bin/bash
set -x
# Be very wary of this explicit setting of CUDA_VISIBLE_DEVICES. Say you are
# running one task and asked for gres=gpu:1 then setting this variable will mean
# all your processes will want to run GPU 0 - disaster!! Setting this variable
# only makes sense in specific cases that I have described above where you are
# using gres=gpu:8 and I have spawned 8 tasks. So I need to divvy up the GPUs
# between the tasks. Think THRICE before you set this!!
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
echo $SLURM_NTASKS
echo $SLURM_LOCALID
rm /tmp/rndv/
mkdir -p /tmp/rndv
BUILD_DIR=/scratch/slurm_tmpdir/$SLURM_JOB_ID/$1
AF_MAX_BUFFERS=3000 $BUILD_DIR/flashlight/build/examples/Resnet34 --world_rank $SLURM_LOCALID --world_size $SLURM_NTASKS --rndv_filepath /tmp/rndv/

# Your CUDA enabled program here
