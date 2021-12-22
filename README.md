# SimSiam-VQ

Course project for IFT 6268 at UdeM. This repo is a fork of [SimSiam-91.9-top1-acc-on-CIFAR10](https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10).

## Intructions

Training on CIFAR-10:

```sh
python main.py --data_root "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log" \
    --trial 'type=vq-lr=0.02-epochs=800-bs=512-K-256-D=64-upscale=1-tau=1.0-tau_end=800-vq_weight=1.0-beta=0.25-n_conv=2-fix_lr=False-fix_tau=False-init-res_proj-grad_norm' \
    --arch resnet18vq --learning_rate 0.02 --epochs 800 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --eval_freq 5 \
    --beta 0.25 --num_embeddings 256 --vq_loss_weight 1.0 --embedding_dim 64 --upscale_factor 1 \
    --tau 1.0 --tau_schedule_end 800 --discrete_type 'vq' \
    --n_proj_conv 2 --res_proj
```

Linear evaluation:

```sh
python main_lincls.py "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log/$TRIAL_NAME" --arch resnet18 --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 --pretrained "$DATA_DIR/log/$TRIAL_NAME/${TRIAL_NAME}_last.pth" \
    --num_embeddings 256 --tau 0.0625 --embedding_dim 64 --discrete_type 'vq'
```
