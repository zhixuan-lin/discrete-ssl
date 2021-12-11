DATA_DIR="/network/scratch/z/zhixuan.lin/discrete-ssl"
TRIAL_NAME="discrete-gumbel-tau_1.0-beta_0.25-vq_weight_1.0-K_256-lr_0.01-reset_prob_0.0-D_512-epochs_1600-tau_schedule_1600-mse-conv_mlp_proj-raw-vq"
mkdir -p $DATA_DIR

python main_lincls.py "$DATA_DIR/CIFAR" --exp_dir "$DATA_DIR/log/$TRIAL_NAME" --arch resnet18 --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 --pretrained "$DATA_DIR/log/$TRIAL_NAME/${TRIAL_NAME}_best.pth" --num_embeddings 256 --tau 0.0625 --embedding_dim 512 --vq
