# Experiment setting

* style_shoes_label_folder: 原本模型架構
* style_shoes_label_folder2: 將原本兩個gen改成一個
* style_shoes_label_folder_v2 : 一個 gen + cycle  loss*10 (MASTER)

## randn --> style encoder

* style_shoes_label_folder_v3 : style code 沒有使用 randn取 (MASTER_v2)
  * model = MASTER_v2
* style_shoes_label_folder_v4 [241]: style code 沒有使用 randn取 + cycle loss *10 (MASTER_v2)
  * model = MASTER_v2 + cycle  loss 
* style_shoes_label_folder_v4 [112]: add flowing on one generator with randn noise
  * model = MASTER + flowing
* style_shoes_label_folder_v5 : 
  * model = MASTER_v2 + flowing
## commmand 

- nohup python master_train.py --config configs/style_shoes_label_folder_v2.yaml --trainer MASTER_v2 --gpu 0 &> v2_log.txt &