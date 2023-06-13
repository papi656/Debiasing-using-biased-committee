DATA="MedMentions"
WEIGHT_FUNC="linear_reweight"


echo "Performance on $DATA after epoch 0"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_1epoch.txt 

echo "Performance on $DATA after epoch 1"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_2epoch.txt 

echo "Performance on $DATA after epoch 2"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_3epoch.txt 

echo "Performance on $DATA after epoch 3"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_4epoch.txt 

echo "Performance on $DATA after epoch 4"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_5epoch.txt 

echo "Performance on $DATA after epoch 5"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_6epoch.txt 

echo "Performance on $DATA after epoch 6"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_7epoch.txt 

echo "Performance on $DATA after epoch 7"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_8epoch.txt 

echo "Performance on $DATA after epoch 8"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_9epoch.txt 

echo "Performance on $DATA after epoch 9"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_10epoch.txt 

echo "Performance on $DATA after epoch 10"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_11epoch.txt 

echo "Performance on $DATA after epoch 11"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_12epoch.txt 

echo "Performance on $DATA after epoch 12"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_13epoch.txt 

echo "Performance on $DATA after epoch 13"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_14epoch.txt 

echo "Performance on $DATA after epoch 14"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_15epoch.txt 

echo "Performance on $DATA after epoch 15"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_16epoch.txt 

echo "Performance on $DATA after epoch 16"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_17epoch.txt 

echo "Performance on $DATA after epoch 17"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_18epoch.txt 

echo "Performance on $DATA after epoch 18"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_19epoch.txt 

echo "Performance on $DATA after epoch 19"
echo ""
python evaluate.py --mention_dictionary datasets/$DATA/mention_dict.txt --cui_dictionary datasets/$DATA/cui_dict.txt --gold_labels datasets/$DATA/test.txt  --gold_cuis datasets/$DATA/test_cuis.txt --predictions resources/$DATA/$WEIGHT_FUNC/preds_clf_20epoch.txt 
