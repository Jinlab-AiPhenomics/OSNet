## Train
The training process is same as MMdetection training process. Please see [mmdetection](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html) for details.
```
# one GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multiple GPUs traing
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
## Test
Most testing process is same as the MMdetection testing process. Please see [mmdetection](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html) for details.
```
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
