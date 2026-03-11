
nvidia-smi

device=0



python main_seg.py --exp_name='E1' \
--data_path='./data/IR_images_sub' \
--local_model='UNet2D' \
--dataset='Meibomian_Gland' \
--T=100 \
--E=4 \
--method='FedLWS' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05  >> output_Gland_FedLWS.txt


