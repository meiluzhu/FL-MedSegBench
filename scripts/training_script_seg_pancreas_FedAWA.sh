
nvidia-smi

device=2

python main_seg.py --exp_name='E1' \
--data_path='./data/Pancreas_segmentation/t1' \
--local_model='SANet' \
--dataset='Pancreas' \
--T=200 \
--E=1 \
--method='FedAWA' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/Pancreas_segmentation/t1' \
--local_model='SANet' \
--dataset='Pancreas' \
--T=100 \
--E=2 \
--method='FedAWA' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/Pancreas_segmentation/t1' \
--local_model='SANet' \
--dataset='Pancreas' \
--T=50 \
--E=4 \
--method='FedAWA' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt