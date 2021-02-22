rm -rf core.* 

# Set up  dataset root dir
DATA_ROOT=/DATA/disk1/ImageNet/ofrecord

# Set up model path, e.g. :  vgg16_of_best_model_val_top1_721  alexnet_of_best_model_val_top1_54762 

MODEL_LOAD_DIR="./output/snapshots/model_save-20210130142727/snapshot_epoch_159"
# MODEL_LOAD_DIR="/home/zzk/oneflow/repvgg/Classification/cnns/test_unzip_model/output/snapshots/model_save-20210129110848/snapshot_epoch_119"


  python3  of_cnn_evaluate.py \
    --num_epochs=3 \
    --num_val_examples=50000 \
    --model_load_dir=$MODEL_LOAD_DIR  \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --node_ips='127.0.0.1' \
    --gpu_num_per_node=4 \
    --val_batch_size_per_device=64 \
    --model="repvggA0"
