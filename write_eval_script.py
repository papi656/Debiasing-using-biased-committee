epochs = 20
with open('bc5cdr_linear_eval.sh', 'w') as fh:
    fh.write(f'DATA="BC5CDR"\n')
    fh.write(f'WEIGHT_FUNC="linear_reweight"\n\n')
    for i in range(epochs):
        fh.write(f'\necho "Performance on $DATA after epoch {i}"\n')
        fh.write(f'echo ""\n')
        fh.write(f'python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_{i+1}epoch.txt \n')
