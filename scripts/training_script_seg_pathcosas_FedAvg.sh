
nvidia-smi

device=0

python main_seg.py --exp_name='E1' \
--data_path='./data/COSAS24/task2' \
--local_model='UNet2D' \
--dataset='Pathology_COSAS2024' \
--T=400 \
--E=1 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> outut_COSA_FedAvg8.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/COSAS24/task2' \
--local_model='UNet2D' \
--dataset='Pathology_COSAS2024' \
--T=200 \
--E=2 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> outut_COSA_FedAvg.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/COSAS24/task2' \
--local_model='UNet2D' \
--dataset='Pathology_COSAS2024' \
--T=100 \
--E=4 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> outut_COSA_FedAvg.txt


python main_seg.py --exp_name='E1' \
--data_path='./data/COSAS24/task2' \
--local_model='UNet2D' \
--dataset='Pathology_COSAS2024' \
--T=50 \
--E=8 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> outut_COSA_FedAvg.txt