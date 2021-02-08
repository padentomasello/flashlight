#!/bin/bash
set -x
set -e
sha=$1
args="${@:2}"
output=$(sbatch -H $args batch_submit_detr.sh $sha)
echo $output
tag=${output#Submitted batch job}
git tag $tag $sha
git push --tags me
scontrol release $tag

