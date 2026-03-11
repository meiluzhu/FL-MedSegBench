
nvidia-smi

device=0

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=200 \
--E=1 \
--method='FedLWS' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=100 \
--E=2 \
--method='FedLWS' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=50 \
--E=4 \
--method='FedLWS' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--beta=0.03 \
--min_tau=0.01 \
--max_tau=0.05 \
--loss='dice_bce' >> output.txt
