DATA_DIR="/network/scratch/z/zhixuan.lin/discrete-ssl"
TRIAL_NAME="type=vq-lr=0.06-epochs=800-bs=512-K-256-D=64-upscale=1-tau=1.0-tau_end=800-vq_weight=1.0-beta=0.25-n_conv=2-fix_lr=False-fix_tau=False"
mkdir -p $DATA_DIR

python main_lincls.py "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log/$TRIAL_NAME" --arch resnet18 --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 --pretrained "$DATA_DIR/log/$TRIAL_NAME/${TRIAL_NAME}_best.pth" \
    --num_embeddings 256 --tau 0.0625 --embedding_dim 64 --discrete_type 'vq'
