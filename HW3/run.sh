python train.py --epochs 100 --batch_size 128 --lr 2e-4 --timesteps 1000 --save ddpm_mnist.pt

python generate.py --ckpt ddpm_mnist.pt --out_dir gen_imgs --num 10000 --batch_size 256 --make_grid --zip_name img_R13942126.zip

python -m pytorch_fid ./gen_imgs ./mnist
# python -m pytorch_fid ./gen_imgs ./mnist.npz