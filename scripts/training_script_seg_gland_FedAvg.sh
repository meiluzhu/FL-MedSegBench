
nvidia-smi

device=0


python main_seg.py --exp_name='E1' \
--data_path='./data/IR_images_sub' \
--local_model='UNet2D' \
--dataset='Meibomian_Gland' \
--T=400 \
--E=1 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> output_Gland_FedAvg.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/IR_images_sub' \
--local_model='UNet2D' \
--dataset='Meibomian_Gland' \
--T=200 \
--E=2 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> output_Gland_FedAvg.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/IR_images_sub' \
--local_model='UNet2D' \
--dataset='Meibomian_Gland' \
--T=100 \
--E=4 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> output_Gland_FedAvg.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/IR_images_sub' \
--local_model='UNet2D' \
--dataset='Meibomian_Gland' \
--T=50 \
--E=8 \
--method='FedAvg' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce'  >> output_Gland_FedAvg.txt


