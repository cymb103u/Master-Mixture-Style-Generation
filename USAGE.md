![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
## Domain Flow for Mixture Style Generation on Latent Space Exploration and Control
## 流域應用於隱空間探索與控制完成混合風格生成
--------------------------------------------------------------------------------
### Dependency
pytorch, yaml, tensorboard (from https://github.com/dmlc/tensorboard), and tensorboardX (from https://github.com/lanpa/tensorboard-pytorch).


The code base was developed using Anaconda with the following packages.
```
pip install tensorboard tensorboardX;
```

### Example Usage

####  Description
- master_train.py : 使用這個檔案來執行訓練；如果只想進行 Disentangle 的話，可把Flowing 部分 mark 掉執行。
- master_trainer.py : 整個模型架構的組合，以及loss function的設定。主要是用 `MASTER_Trainer`。
- master_networks.py : 網路的component。
- master_test.py : 在 inference 時使用。
- utils .py : 工具function
- Note.md : 實驗記錄的部分 & 訓練command
  
#### Training
1. Prepare dataset
   + 將 dataset 分成 trainA / trainB 和 testA/testB，放到 `datasets`

2. Setup the yaml file. 
   + Check out `configs/style_shoes_label_floder_OO.yaml` for folder-based dataset organization. 

3. 開啟 Visdom  `python -m visdom.server`
   
4. Start training
    ```
    python master_train.py --config configs/style_shoes_label_folder_OO.yaml --trainer MASTER
    ```
    
5. Intermediate image outputs and model binary files are stored in `outputs/style_shoes_label_floder_OO`


#### Testing 

First, download the [pretrained models](https://drive.google.com/drive/folders/10IEa7gibOWmQQuJUIUOkh-CV4cm6k8__?usp=sharing) and put them in `models` folder.

 
