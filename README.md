# HRCenterNet (Official Pytorch Implementation)
Chinese Character Detection in Historical Documents
![results](https://github.com/Tverous/HRCenterNet/blob/main/images/results.JPG)

#### HRCenterNet: An Anchorless Approach to Chinese Character Segmentation in Historical Documents https://arxiv.org/abs/2012.05739
IEEE Big Data 2020, CAS Workshop
Chia-Wei Tang, Chao-Lin Liu, Po-Sen Chu

Department of Computer Science, National Chengchi University

## Installation
```
git clone https://github.com/Tverous/HRCenterNet.git
cd HRCenterNet/
```
## Download pretrained weight

[Google Drive](https://drive.google.com/file/d/1EM00B9mh9jb8byEl0vLFtcfF_FdI65SH/view?usp=sharing)

## How to use ?
- Test with images:

  `python test.py --data_dir /path/to/images --log_dir /path/to/pretrained --output_dir /path/to/save/outputs`

## Results
![results_1](https://github.com/Tverous/HRCenterNet/blob/main/images/results_1.png)
![results_2](https://github.com/Tverous/HRCenterNet/blob/main/images/results_2.png)


## Todo
- [ ] Complete README
- [ ] Release training data and make training code more readable
