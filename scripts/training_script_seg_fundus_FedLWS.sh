
nvidia-smi

device=0


python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Fundus_Vessel_Segmentation' \
--local_model='UNet2D' \
--dataset='Fundus' \
--T=400 \
--E=1 \
--method='FedLWS' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output_Fundus_FedLWS.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Fundus_Vessel_Segmentation' \
--local_model='UNet2D' \
--dataset='Fundus' \
--T=200 \
--E=2 \
--method='FedLWS' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output_Fundus_FedLWS.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Fundus_Vessel_Segmentation' \
--local_model='UNet2D' \
--dataset='Fundus' \
--T=100 \
--E=4 \
--method='FedLWS' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output_Fundus_FedLWS.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Fundus_Vessel_Segmentation' \
--local_model='UNet2D' \
--dataset='Fundus' \
--T=50 \
--E=8 \
--method='FedLWS' \
--batchsize=8 \
--lr=0.001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output_Fundus_FedLWS.txt

