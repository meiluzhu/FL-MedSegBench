#!/bin/sh


#SBATCH --job-name=ID45
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1


echo "Submission Directory : " $SLURM_SUBMIT_DIR
echo "Submission Host      : " $SLURM_SUBMIT_HOST
echo "Job User             : " $SLURM_JOB_USER
echo "Job ID               : " $SLURM_JOB_ID
echo "Job Name             : " $SLURM_JOB_NAME
echo "Queue                : " $SLURM_JOB_PARTITION
echo "Node(s) allocated    : " $SLURM_JOB_NODELIST
echo "Number of Node(s)    : " $SLURM_NNODES
echo "Number of CPU Task(s): " $SLURM_NTASKS
echo "Number of Process(s) : " $SLURM_NPROCS
echo "Task(s) per Node     : " $SLURM_TASKS_PER_NODE
echo "CPU(s) per Task      : " $SLURM_CPUS_PER_TASK
echo "Task ID              : " $SLURM_ARRAY_TASK_ID

echo ===========================================================
echo "Job Start  Time is `date "+%Y/%m/%d -- %H:%M:%S"`"

cd $WORK
OUTFILE=${SLURM_JOB_NAME}.${SLURM_JOB_ID}


nvidia-smi

device='0'


python $SLURM_SUBMIT_DIR/main_seg.py --exp_name='E1' \
--data_path='./data/MICCAI_FeTS2022_TrainingData' \
--local_model='UNet3D' \
--dataset='FeTS2022' \
--T=50 \
--E=4 \
--method='FedRoD' \
--batchsize=4 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'
