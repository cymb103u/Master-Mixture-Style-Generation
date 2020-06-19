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
* style_shoes_label_folder_v6 : 在decode的時候加入domain資訊
  * model = MASTER + flowing
* style_shoes_label_folder_v7 : 將 iteration 改成500000 和 domainess sample on uniform distribution
## commmand 

- nohup python master_train.py --config configs/style_shoes_label_folder_v2.yaml --trainer MASTER_v2 --gpu 0 &> v2_log.txt &
- nohup python master_train.py --config configs/style_shoes_label_folder_v7.yaml --trainer MASTER --gpu 1 --port 8000 &> v7_log.txt &
- python master_test.py --trainer MASTER --config configs/style_shoes_label_folder_v4.yaml --checkpoint outputs/style_shoes_label_folder_v4/checkpoints/gen_00500000.pt