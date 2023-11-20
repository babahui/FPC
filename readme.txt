1. Requirements Installation
conda env create -f env.yaml

2. Prepare Datasets
Download the Occluded or Holistic Person ReID datasets (e.g. Occluded-Duke), Then unzip them and rename them under the './data' directory.

3. Prepare ViT Pre-trained Models
Download Pre-trained ViT Models in "https://drive.google.com/file/d/1UPiHFjdhMYqADs4I6THok8zhwhe_l21V/view?usp=share_link", and put in directory as './jx_vit_base_p16_224-80ecf9dd.pth'.
Download FPC Models in "https://drive.google.com/file/d/1DrgUzoUTpiZLpPrWiQyz2_QeZ5z1GfXj/view?usp=share_link", and put in directory like './FPC_Occ_Duke_reconstruction.pth'

4. Training & Evaluation
Pelase refer to 'train&test.ipynb' for evaluation.

Some Results of Occluded-Duke:
2023-03-15 15:20:51,581 transreid.test INFO: Validation Results 
2023-03-15 15:20:51,581 transreid.test INFO: mAP: 72.7%
2023-03-15 15:20:51,581 transreid.test INFO: CMC curve, Rank-1  :76.8%
2023-03-15 15:20:51,581 transreid.test INFO: CMC curve, Rank-5  :83.0%
2023-03-15 15:20:51,581 transreid.test INFO: CMC curve, Rank-10 :86.9%