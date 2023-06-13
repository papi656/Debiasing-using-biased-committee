EPOCHS=2
ENSEMBLE_SZ=96
DATA="BC5CDR"

echo ""
echo "Committee warmup training"
echo "Dataset: $DATA"
echo "Ensemble size: $ENSEMBLE_SZ"
echo "Epochs: $EPOCHS"
echo ""

python bias_comm_warmup.py \
    --dataset_name $DATA \
    --model_name model \
    --ensemble_size $ENSEMBLE_SZ \
    --epochs $EPOCHS