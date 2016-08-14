export LD_LIBRARY_PATH=$HOME/local/cuda-7.5/lib64:$HOME/local/cudnn-7.5-v5.0.5/lib64

CURD=`pwd`

cd ../../../

python -m alchemy.lab.deepid.train \
  --network deepid \
  --data-dir=$CURD/data \
  --save-model-prefix=$CURD/snapshot/deepid \
  --lr=0.05 \
  --lr-factor=0.9 \
  --lr-factor-epoch=1 \
  --num-epochs=100 \
  --batch-size=512 \
  --use-bn=True \
  --data-shape=55,55 \
  --gpus=1,2,3 \
  --num-examples=31900 \
  --num-classes=1595 \
  --log-dir=$CURD/logs \
  --log-file=deepid.log \
  --train-dataset=train.rec \
  --val-dataset=val.rec \
  --metrics=top1,ce \
  --num-batch-callback=50
