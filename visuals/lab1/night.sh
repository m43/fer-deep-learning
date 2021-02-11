# --npl 784 100 10
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 10000 --es 10000 --wd 0 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 10000 --es 10000 --wd 1e-4 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 10000 --es 10000 --wd 1e-3 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 10000 --es 10000 --wd 1e-2 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 10000 --es 10000 --wd 1 --eta 0.1 --seed 72

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 20000 --es 20000 --wd 1e-3 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --epochs 20000 --es 20000 --wd 1e-2 --eta 0.1 --seed 72

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1000 --log_images_interval 400000 --v 1 --epochs 100000 --es 1000 --wd 1e-3 --eta 0.1 --seed 72

# 0     72 192 360
# 1e-4  72 192 360
# 1e-3  72 192 360
# 1e-2  72 192 360
# 1     72 192 360

# --npl 784 100 100 10
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 0 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 0 --eta 0.1 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 0 --eta 0.1 --seed 360

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-4 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-4 --eta 0.1 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-4 --eta 0.1 --seed 360

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-3 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-3 --eta 0.1 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-3 --eta 0.1 --seed 360

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-2 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-2 --eta 0.1 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1e-2 --eta 0.1 --seed 360

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1 --eta 0.1 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 10 --epochs 100000 --es 1000 --wd 1 --eta 0.1 --seed 360

#--npl 784 100 100 100 10
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 0 --eta 0.01 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 1e-4 --eta 0.01 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-3 --eta 0.03 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-2 --eta 0.1 --seed 72
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1 --eta 0.03 --seed 72

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 0 --eta 0.01 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 1e-4 --eta 0.01 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-3 --eta 0.03 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-2 --eta 0.03 --seed 192
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1 --eta 0.03 --seed 192

python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 0 --eta 0.01 --seed 360
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 1 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 200000 --es 1000 --wd 1e-4 --eta 0.01 --seed 360
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-3 --eta 0.03 --seed 360
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1e-2 --eta 0.03 --seed 360
python.exe -m demo.mlp.mnist_shootout --init kaiming_uniform --log_interval 100 --log_images_interval 400000 --v 1 --npl 784 100 100 100 10 --epochs 300000 --es 2000 --wd 1 --eta 0.03 --seed 360
