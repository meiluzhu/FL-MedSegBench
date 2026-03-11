
nvidia-smi

device=0

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Breast_Ultrasound' \
--local_model='UNet2D' \
--dataset='FL_Breast_Ultrasound' \
--T=400 \
--E=1 \
--method='FedRDN' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_Ultrasound_FedRDN.txt


python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Breast_Ultrasound' \
--local_model='UNet2D' \
--dataset='FL_Breast_Ultrasound' \
--T=200 \
--E=2 \
--method='FedRDN' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_Ultrasound_FedRDN.txt


python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Breast_Ultrasound' \
--local_model='UNet2D' \
--dataset='FL_Breast_Ultrasound' \
--T=100 \
--E=4 \
--method='FedRDN' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_Ultrasound_FedRDN.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_Breast_Ultrasound' \
--local_model='UNet2D' \
--dataset='FL_Breast_Ultrasound' \
--T=50 \
--E=8 \
--method='FedRDN' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_Ultrasound_FedRDN.txt

