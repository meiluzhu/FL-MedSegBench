
nvidia-smi

device=0



python main_seg.py --exp_name='E1' \
--data_path='./data/COSAS24/task2' \
--local_model='UNet2D' \
--dataset='Pathology_COSAS2024' \
--T=100 \
--E=4 \
--method='FedAWA' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_COSAS_FedAWA.txt


