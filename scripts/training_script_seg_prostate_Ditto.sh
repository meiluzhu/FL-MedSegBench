
nvidia-smi

device=0

python main_seg.py --exp_name='E1' \
--data_path='./data/Multi_site_Prostate' \
--local_model='UNet2D' \
--dataset='Prostate' \
--T=400 \
--E=1 \
--method='Ditto' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_prostate_Ditto.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/Multi_site_Prostate' \
--local_model='UNet2D' \
--dataset='Prostate' \
--T=200 \
--E=2 \
--method='Ditto' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_prostate_Ditto.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/Multi_site_Prostate' \
--local_model='UNet2D' \
--dataset='Prostate' \
--T=100 \
--E=4 \
--method='Ditto' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_prostate_Ditto.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/Multi_site_Prostate' \
--local_model='UNet2D' \
--dataset='Prostate' \
--T=50 \
--E=8 \
--method='Ditto' \
--batchsize=8 \
--lr=0.0001 \
--num_classes=1 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output_prostate_Ditto.txt


