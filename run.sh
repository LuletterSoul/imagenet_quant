# CUDA_VISIBLE_DEVICES=3 python main_mobv2.py --data datasets/imagenet --pretrained -e -b 256 -j 16 --output_dir outputs/mobv2 
python main_mobv2.py --data datasets/imagenet --pretrained -e -b 1024 -j 64 --output_dir outputs/mobv2_grid_search --dist-backend 'nccl' --dist-url 'tcp://127.0.0.1:6006
' --multiprocessing-distributed --world-size 1 --rank 0 -q --quant_layer features.0.0 --quant_rename_layer conv1
# python main_mobv2.py --data datasets/imagenet --pretrained -e -b 1024 -j 64 --output_dir outputs/mobv2_distributed  -q --world-size 1