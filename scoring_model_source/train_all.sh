#!/bin/bash

LANG="en"
LABEL_TYPE1="pron"
LABEL_TYPE2="articulation"
DIR_DATA="data/audio"
DIR_LIST="/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_small_list"
MLP_HIDDEN=64
EPOCHS=200
PATIENCE=20
BATCH_SIZE=256
DIR_MODEL="model_test_pron+articulation"

for MODEL_TYPE in "mlp" "cnn" "lstm" "transformer"; do
    python train_mlp.py \
        --lang="$LANG" \
        --label_type1="$LABEL_TYPE1" \
        --label_type2="$LABEL_TYPE2" \
        --dir_data="$DIR_DATA" \
        --dir_list="$DIR_LIST" \
        --mlp_hidden="$MLP_HIDDEN" \
        --epochs="$EPOCHS" \
        --patience="$PATIENCE" \
        --batch_size="$BATCH_SIZE" \
        --dir_model="$DIR_MODEL" \
        --model_type="$MODEL_TYPE"
done


# python train_mlp\ copy.py --lang='en' --label_type1='pron' --label_type2='articulation' --dir_data='data/audio' --dir_list='/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_sample' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model_test1' --model_type='mlp'
# python train_mlp\ copy.py --lang='en' --label_type1='pron' --label_type2='articulation' --dir_data='data/audio' --dir_list='/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_sample' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model_test1' --model_type='cnn'
# python train_mlp\ copy.py --lang='en' --label_type1='pron' --label_type2='articulation' --dir_data='data/audio' --dir_list='/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_sample' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model_test1' --model_type='lstm'
# python train_mlp\ copy.py --lang='en' --label_type1='pron' --label_type2='articulation' --dir_data='data/audio' --dir_list='/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_sample' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model_test1' --model_type='transformer'