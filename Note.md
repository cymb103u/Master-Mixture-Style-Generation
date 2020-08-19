# Experiment setting

* style_shoes_label_folder: 原本模型架構
* style_shoes_label_folder2: 將原本兩個gen改成一個
* w/o flowing
  * style_shoes_label_folder_v2 : 一個 gen + cycle  loss*10 (MASTER) // 20/07/21 [112]
  * style_shoes_label_folder_v11 : 一個 gen + w/o latent loss // 20/07/15 [112]
  * style_shoes_label_folder_v12 :  一個 gen + w/o reconstruction loss   20/07/15 [241]
* with flowing
  * style_shoes_label_folder_v9 : iteration 500000 + beta distribution(half) + flowing加上content loss (Good) [241]
## randn --> style encoder

* style_shoes_label_folder_v3 : style code 沒有使用 randn取 (MASTER_v2)
  * model = MASTER_v2
* style_shoes_label_folder_v4 [241]: style code 沒有使用 randn取 + cycle loss *10 (MASTER_v2)
  * model = MASTER_v2 + cycle  loss 
* style_shoes_label_folder_v4 [112]: add flowing on one generator with randn noise
  * model = MASTER + flowing
* style_shoes_label_folder_v5 : 
  * model = MASTER_v2 + flowing
* style_shoes_label_folder_v6 : 在decode的時候加入domain資訊 + iteration 改成500000 +  beta distribution
  * model = MASTER + flowing
* style_shoes_label_folder_v7 : 將 iteration 改成500000 和 domainess sample on uniform distribution
* style_shoes_label_folder_v8 : 在decode的時候加入domain資訊 + iteration 1000000 +  beta distribution
  * model = MASTER + flowing
* style_shoes_label_folder_v9 : iteration 500000 + beta distribution(half) + flowing加上content loss (Good) [在241上]
* style_shoes_label_folder_v10 : 跟v9一樣+ lerp + + 更改過後flowing loss

* style_shoes_label_folder_v13 : 重現v9
## commmand 

- nohup python master_train.py --config configs/style_shoes_label_folder_v2.yaml --trainer MASTER_v2 --gpu 0 &> v2_log.txt &
- nohup python master_train.py --config configs/style_shoes_label_folder_v7.yaml --trainer MASTER --gpu 1 --port 8000 &> v7_log.txt &
- python master_test.py --trainer MASTER --config configs/style_shoes_label_folder_v4.yaml --checkpoint outputs/style_shoes_label_folder_v4/checkpoints/gen_00500000.pt
- windows 
  - start/min python master_train.py --config configs/style_shoes_label_folder_v8.yaml --trainer MASTER --gpu 1 --port 8097

## Diversity
- all_real_test_data : Avg: 0.36281 +/- 0.00323
-  全部合成去算 LPIPS

## visdom 監控
- ssh -L 127.0.0.1:6006:127.0.0.1:6006 cymb103u@140.123.104.112
 