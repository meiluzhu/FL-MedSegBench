
nvidia-smi

device=0

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=200 \
--E=1 \
--method='FedNova' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=100 \
--E=2 \
--method='FedNova' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt

python main_seg.py --exp_name='E1' \
--data_path='./data/FL_MMS' \
--local_model='UNet3D' \
--dataset='MMS' \
--T=50 \
--E=4 \
--method='FedNova' \
--batchsize=2 \
--lr=0.0001 \
--num_classes=4 \
--device=$device \
--optimizer='adam' \
--loss='dice_bce' >> output.txt
