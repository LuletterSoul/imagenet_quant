# CUDA_VISIBLE_DEVICES=3 python main_mobv2.py --data datasets/imagenet --pretrained -e -b 256 -j 16 --output_dir outputs/mobv2 
# Q_DIV =("" "r22" "r32" "r42")
BITs = ("8")
# Q_DIVs = (127, 127.1, 127.2, 127.3, 127.4, 127.5, 127.6, 127.7, 127.8, 127.9, 128, 128.1, 128.5, 200)
Q_DIVs = ("127")
# Q_DIVs = ("127", "127.1","127.2", "127.3", "127.4", "127.5", "127.6", "127.7", "127.8", "127.9", "128", "128.1", "128.5", "129", "200")
for BIT in ${BITs[@]};
    do
	for Q_DIV in ${Q_DIVs[@]};
	  do	 	

            python main_mobv2.py --data datasets/imagenet --pretrained -e -b 1024 -j 64 --output_dir outputs/mobv2_grid_search --dist-backend 'nccl' --dist-url 'tcp://127.0.0.1:6006
            ' --multiprocessing-distributed --world-size 1 --rank 0 -q --BIT $BIT --Q_DIV $Q_DIV
          done
    done

# python main_mobv2.py --data datasets/imagenet --pretrained -e -b 1024 -j 64 --output_dir outputs/mobv2_distributed  -q --world-size 1