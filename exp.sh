#!/bin/bash
#SBATCH --job-name=nilmjob
#SBATCH --account=da33
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40000
#SBATCH --gres=gpu:2
#SBATCH --partition=m3h
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurrm-%j.out
# SBATCH --error=logs/slurrm-%j.err
# load the module 
module load  tensorflow/1.4.0-python3.6-gcc5

cd /projects/da33/ozeidi/Project/experiments/e104
python src/experiment.py $1 $2 $3
# python src/experiment.py kettle 0 5
#python src/experiment.py fridge 0 10
# python src/experiment.py dish_washer 0 5
# python src/experiment.py washing_machine 0 5
#python src/experiment.py microwave 0 10