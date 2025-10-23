#!/bin/bash
# Start TensorBoard without pkg_resources warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate myenv

# Start TensorBoard
# tensorboard --logdir="logs/exp_2025-10-18 21:13:51" --port 6011 #SimpleNEt Modified
# tensorboard --logdir="logs/exp_2025-10-17 19:27:27" --port 6010 #Resnet 18
# tensorboard --logdir="logs/exp_2025-10-16 19:03:37" --port 6009 #Pytorch CNN
# tensorboard --logdir="logs/exp_2025-10-16 18:19:46" --port 6008 #Custom Conv
tensorboard --logdir="logs/exp_2025-10-22 15:37:31" --port 6006 #VIT