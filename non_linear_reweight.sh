WEIGHT_FUNC="non_linear_reweight"
ENSEMBLE_SZ=96
DATA="BC5CDR"
EPOCHS=10

if [ ! -d resources/${DATA}/${WEIGHT_FUNC} ];
then
    mkdir resources/${DATA}/${WEIGHT_FUNC}
fi 

# using linear weighting
python weighted_training.py \
    --dataset_name $DATA \
    --model_name model \
    --ensemble_size $ENSEMBLE_SZ \
    --weight_func $WEIGHT_FUNC \
    --num_epochs $EPOCHS

