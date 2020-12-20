# HRCenterNet (Official Pytorch Implementation)
Chinese Character Detection in Historical Documents
![results](https://github.com/Tverous/HRCenterNet/blob/main/images/results.JPG)

[HRCenterNet: An Anchorless Approach to Chinese Character Segmentation in Historical Documents](https://arxiv.org/abs/2012.05739)

Chia-Wei Tang, Chao-Lin Liu, Po-Sen Chu

[IEEE Big Data 2020 Workshops, Computational Archival Science: digital records in the age of big data](https://ai-collaboratory.net/cas/cas-workshops/ieee-big-data-2020-5th-cas-workshop/)

## Installation
```
git clone https://github.com/Tverous/HRCenterNet.git
cd HRCenterNet/
pip install -r requirements.txt
```
## Download pretrained weight

[Google Drive - hrcenternet.pth](https://drive.google.com/file/d/1EM00B9mh9jb8byEl0vLFtcfF_FdI65SH/view?usp=sharing)

## Download Dataset

[Google Drive - train.csv](https://drive.google.com/file/d/1wRRDhILEBfOO3CKT32M0AXUp4iINdRbF/view?usp=sharing)

[Google Drive - val.csv](https://drive.google.com/file/d/1W2DgwUFlrUjJiWCGXiBk7pnJk8rZwmbQ/view?usp=sharing)

[Google Drive - dataset_images.zip](https://drive.google.com/file/d/1syj7Osi0ACqbuuhkoZsuOWXW7Gtjov05/view?usp=sharing)

## How to use ?
- Training:
  ```
  python train.py --train_csv_path data/train.csv --train_data_dir data/images \
                  --val_csv_path data/val.csv --val_data_dir data/images/ --val \
                  --batch_size 8 --epoch 80
  ```
- Evaluation:
  ```
  python evaluate.py --csv_path data/val.csv --data_dir data/images/ --log_dir weights/hrcenternet.pth
  ```
- Testing:
  ```
  python test.py --data_dir /path/to/images --log_dir /path/to/pretrained --output_dir /path/to/save/outputs
  ```

## Training on Your Own Dataset
Prepare your csv files with following format:

```
image_id              labels
file_name_1           object_1 center_x center_y width height object_2 center_x center_y width height ...
file_name_2           object_1 center_y center_y width height object_2 center_x center_y width height ...
    .                 .
    .                 .
    .                 .
```

## Results
![results_1](https://github.com/Tverous/HRCenterNet/blob/main/images/results_1.png)
![results_2](https://github.com/Tverous/HRCenterNet/blob/main/images/results_2.png)

## Citation
Use this bibtex to cite this repository:
```
@misc{2012.05739,
  Author = {Chia-Wei Tang, Chao-Lin Liu and Po-Sen Chiu},
  Title = {HRCenterNet: An Anchorless Approach to Chinese Character Segmentation in Historical Documents},
  Year = {2020},
  Eprint = {arXiv:2012.05739},
}
```
