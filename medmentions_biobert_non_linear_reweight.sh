WEIGHT_FUNC="non_linear_reweight"
ENSEMBLE_SZ=96
DATA="MedMentions"
EPOCHS=20

echo ""
echo "Training using $WEIGHT_FUNC"
echo "Dataset: $DATA"
echo "Ensemble size: $ENSEMBLE_SZ"
echo "Epochs: $EPOCHS"
echo ""

if [ ! -d resources/${DATA}/${WEIGHT_FUNC} ];
then 
    mkdir resources/${DATA}/${WEIGHT_FUNC} 
fi 

python weighted_biobert_training.py \
    --dataset_name $DATA \
    --model_name model \
    --ensemble_size $ENSEMBLE_SZ \
    --weight_func $WEIGHT_FUNC \
    --num_epochs $EPOCHS