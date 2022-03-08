# HRCenterNet (Official Pytorch Implementation)
Chinese Character Detection in Historical Documents
![results](https://github.com/Tverous/HRCenterNet/blob/main/images/results.JPG)

>[HRCenterNet: An Anchorless Approach to Chinese Character Segmentation in Historical Documents](https://arxiv.org/abs/2012.05739) \
>Chia-Wei Tang, [Chao-Lin Liu](https://www.cs.nccu.edu.tw/~chaolin/), Po-Sen Chu \
>[IEEE Big Data 2020 Workshops, Computational Archival Science: digital records in the age of big data](https://ai-collaboratory.net/cas/cas-workshops/ieee-big-data-2020-5th-cas-workshop/) \
> *IEEE Xplore ([10.1109/BigData50022.2020.9378051](https://ieeexplore.ieee.org/document/9378051/metrics#metrics))* \
> *arXiv technical report ([arXiv 2012.05739](https://arxiv.org/abs/2012.05739))*

Contact: [106703054@g.nccu.edu.tw](mailto:106703054@g.nccu.edu.tw). Any questions or discussions are welcomed! 

## Installation
```
git clone https://github.com/Tverous/HRCenterNet.git
cd HRCenterNet/
pip install -r requirements.txt
```

## Download the dataset and the pretrained weight

### Google Drive

- [train.csv](https://drive.google.com/file/d/1wRRDhILEBfOO3CKT32M0AXUp4iINdRbF/view?usp=sharing)

- [val.csv](https://drive.google.com/file/d/1W2DgwUFlrUjJiWCGXiBk7pnJk8rZwmbQ/view?usp=sharing)

- [dataset_images.zip](https://drive.google.com/file/d/1syj7Osi0ACqbuuhkoZsuOWXW7Gtjov05/view?usp=sharing)

- [HRCenterNet.pth.tar](https://drive.google.com/file/d/1pWOZ0M5suplCZeFBJK0SvC34IUtEkOpI/view?usp=sharing)

### OneDrive

- [train.csv](https://1drv.ms/u/s!Aiktfq4vkVE6godo4U135rwRNwG5WA?e=lhvEox)

- [val.csv](https://1drv.ms/u/s!Aiktfq4vkVE6godntw-UrrPt8M2wWw?e=DXuqzZ)

- [dataset_images.zip](https://1drv.ms/u/s!Aiktfq4vkVE6godrkvPEWr8xKFLVNA?e=T1ieS9)

- [HRCenterNet.pth.tar](https://1drv.ms/u/s!Aiktfq4vkVE6godpDqzJb5-DsbEEAg?e=wqQDz2)

## How to use ?
### Training:
  ```
  python train.py --train_csv_path data/train.csv --train_data_dir data/images \
                  --val_csv_path data/val.csv --val_data_dir data/images/ --val \
                  --batch_size 8 --epoch 80
  ```
### Evaluation:
  ```
  python evaluate.py --csv_path data/val.csv --data_dir data/images/ --log_dir weights/HRCenterNet.pth.tar
  ```
### Test with unseen images:
  ```
  python test.py --data_dir /path/to/images --log_dir /path/to/pretrained --output_dir /path/to/save/outputs
  ```

## Training on Your Own Dataset
Prepare your csv files with following format:

```
image_id              labels
file_name_1           obj_id_1 topleft_x topleft_y width height obj_id_2 topleft_x topleft_y width height ...
file_name_2           obj_id_1 topleft_x topleft_y width height obj_id_2 topleft_x topleft_y width height ...
    .                 .
    .                 .
    .                 .
```

## Results
![results_1](https://github.com/Tverous/HRCenterNet/blob/main/images/results_1.png)
![results_2](https://github.com/Tverous/HRCenterNet/blob/main/images/results_2.png)

## Citation
Use this bibtex to cite this repository:
```bibtex
@INPROCEEDINGS{
  9378051,  
  author={C. -W. {Tang} and C. -L. {Liu} and P. -S. {Chiu}},  
  booktitle={2020 IEEE International Conference on Big Data (Big Data)},   
  title={HRCenterNet: An Anchorless Approach to Chinese Character Segmentation in Historical Documents},   
  year={2020},  
  volume={},  
  number={},  
  pages={1924-1930},  
  doi={10.1109/BigData50022.2020.9378051}
}
```
