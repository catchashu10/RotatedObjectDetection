## CS 766 Computer Vision Project

### Introduction
This project aims to adapt the YOLOv7 model to detect arbitrarily oriented objects in remote-sensing images. To achieve this, modifying the original loss function of the model is required. We obtained a successful result by increasing the number of anchor boxes with different rotating angles and combining the smooth-L1-IoU loss function proposed by [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612) into the original loss for generating bounding boxes.
<br><br>


### Features
---
#### Loss Function (only for x, y, w, h, theta)
<img src="https://i.imgur.com/zdA9RJj.png" alt="loss" height="90"/>
<img src="https://i.imgur.com/Qi1XFXS.png" alt="angle" height="70"/>

---

### Setup

1. Clone repository
    ```
    $ git clone https://github.com/catchashu10/RotatedObjectDetection.git
    $ cd R-YOLOv7/
    ```
2. Create Environment

* Conda
    
    1. Create virual environment
        ```
        $ conda create -n ryolo python=3.8
        $ conda activate ryolo
        ```
    2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org), e.g.,
        ```
        If you are using CUDA 11.8 version
        $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        ```
    3. Install required libraries
        ```
        $ pip install -r requirements.txt
        ```
    4. Install detectron2 for calculating SkewIoU on GPU following the [official instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), e.g.,
        ```
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        ```
        
* Docker

    ```
    $ docker build -t ryolo docker/
    $ sudo docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 -v ${your_path}:/workspace ryolo
    ```

6. Download  pretrained weights</br>
    [weights](https://drive.google.com/drive/folders/1PkVYTTN9YlToM3ZbNsUKbAs6RDwh94am?usp=share_link)
    
7. Make sure your files arrangment looks like the following</br>
    Note that each of your dataset folder in `data` should split into three files, namely `train`, `test`, and `detect`.
    ```
    R-YOLOv4/
    ├── train.py
    ├── test.py
    ├── detect.py
    ├── xml2txt.py
    ├── environment.xml
    ├── requirements.txt
    ├── model/
    ├── datasets/
    ├── lib/
    ├── outputs/
    ├── weights/
        ├── pretrained/ (for training)
        └── UCAS-AOD/ (for testing and detection)
    └── data/
        └── UCAS-AOD/
            ├── class.names
            ├── train/
                ├── ...png
                └── ...txt
            ├── test/
                ├── ...png
                └── ...txt
            └── detect/
                └── ...png
    ```
    
### Train

The methods to load and train different datasets, such as UCAS-AOD and DOTA can be found at [/datasets](https://github.com/kunnnnethan/R-YOLOv4/tree/main/datasets). The angle of each bounding box is limited in `(- pi/2,  pi/2]`, and the height of each bounding box is always longer than its width.

Check all the settings in the .yaml files that you will use in the [/data](https://github.com/kunnnnethan/R-YOLOv4/tree/main/data) folder.

```
$ python train.py --model_name DOTA_yolov7_csl_800 --config data/hyp.yaml --img_size 800 --data data/DOTA.yaml --epochs 100 --mode csl --ver yolov7
```

You can run [display_inputs.py](https://github.com/kunnnnethan/R-YOLOv4/blob/main/display_inputs.py) to visualize whether your data is loaded successfully.

#### UCAS-AOD dataset

Please refer to [this repository](https://github.com/kunnnnethan/UCAS-AOD-benchmark) to rearrange files so that the dataset can be loaded and trained by the model.</br>


#### DOTA dataset

Download the official dataset from [here](https://captain-whu.github.io/DOTA/dataset.html). The original files should be able to be loaded and trained by the model.</br>
<br>

### Test
```
$ python test.py --data data/DOTA.yaml --hyp data/hyp.yaml --weight_path weights/DOTA_yolov7_csl_800/best.pth --batch_size 8 --img_size 800 --mode csl --ver yolov7
```

### Detect

```
$ python detect.py --data data/UCAS_AOD.yaml --hyp data/hyp.yaml --weight_path weights/DOTA_yolov7_csl_800/best.pth --batch_size 8 --img_size 800 --conf_thres 0.3 --mode csl --ver yolov7
```

#### Tensorboard
If you would like to use tensorboard for tracking training process.

* Open an additional terminal in the same folder where you are running the program.
* Run command ```$ tensorboard --logdir=weights --port=6006``` 
* Go to [http://localhost:6006/]( http://localhost:6006/)

### Results

#### UCAS_AOD

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0292.png" alt="car" height="430"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0769.png" alt="plane" height="413"/>

#### DOTA

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0006.png" alt="DOTA" height="430"/><img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0031.png" alt="DOTA" height="430"/>


### References

[yingkunwu/R-YOLOv4](https://github.com/yingkunwu/R-YOLOv4)</br></br>

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**

```
@inproceedings{wang2023yolov7,
  title={YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7464--7475},
  year={2023}
}
```

**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**

```
@article{r3det,
  title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
  author={Xue Yang, Junchi Yan, Ziming Feng, Tao He},
  journal = {arXiv},
  year={2019}
}
```
