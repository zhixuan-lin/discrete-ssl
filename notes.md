
Best config:

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

Gumbel-softmax vs VQ

`tau=2.0` scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_2.0' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 2.0
```

`tau=2.0` not scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_2.0-fix_tau' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 2.0 --fix_tau
```

`tau=1.5` scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_1.5' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 1.5
```

`tau=1.5` not scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_1.5-fix_tau' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 1.5 --fix_tau
```

`tau=1.0` scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_1.0' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 1.0
```

`tau=1.0` not scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_1.0-fix_tau' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 1.0 --fix_tau
```

`tau=0.5` scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_0.5' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 0.5
```

`tau=0.5` not scheduled

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_800-tau_schedule_800-mse-conv_mlp_proj-raw-gumbel-tau_0.5-fix_tau' --arch resnet18vq --learning_rate 0.01 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 800 --use_mseloss --conv_mlp_proj --raw --tau 0.5 --fix_tau
```

Beta
```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.05' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.05 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.1' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.1 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 

python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.3' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.3 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.4' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.4 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.5' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.5 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.6' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.6 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_0.75' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.75 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 

python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_1.0' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 1.0 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 

python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq-beta_2.0' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 2.0 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

Codebook size

64

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_64-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 64 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

128

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_128-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 128 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

192

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_192-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 192 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

512

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_512-lr_0.02-reset_prob_0.0-D_512-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 512 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 512 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

Ebedding dim


64
```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_64-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 64 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

128
```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_128-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 128 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

256
```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_256-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 256 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```

384
```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" --trial 'discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.02-reset_prob_0.0-D_384-epochs_800-tau_schedule_1600-mse-conv_mlp_proj-raw-vq' --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --reset_prob 0.0 --embedding_dim 384 --upscale_factor 1 --tau_schedule_end 1600 --use_mseloss --conv_mlp_proj --raw --vq 
```
