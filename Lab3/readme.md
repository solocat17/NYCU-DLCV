# NYCU Computer Vision 2025 Spring HW3

StudentID: 111550089

Name: 李宗諺

## Introduction

In this assignment, we explored Mask R-CNN architecture to tackle a medical cell instance segmentation problem. Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks in parallel with the existing branch for bounding box recognition. I implemented a complete Mask R-CNN pipeline with ResNet-50 as the backbone architecture and explored several modifications to improve performance, including adjusting augmentation strategies, experimenting with resolution scaling, and testing alternative backbones. The model was trained to recognize and segment four different cell types in histopathological images, achieving 0.3253 mAP on the public test dataset despite the inherent challenges of medical image segmentation.

## How to install

It is recommended to use `conda` to manage the environment. The following command will create a new environment and install the required packages.

```bash
conda env create -f environment.yml
```

Then, activate the environment by running

```bash
conda activate dlcv_lab3
```

## How to train
You should place the dataset in the `dataset` folder before training. The only must-have argument is the `output_dir`. To check all arguments, please check `train.py`. You can train the model by running

```bash
python train.py --output_dir result/exp1 \
                --num_epochs 30 \
                --batch_size 2 \
                --lr 0.001 \
                --model_type fpnv2
```

The training result will be saved in the `results/exp1/` folder, along with the weights of the model.

## How to evaluate

There are 2 evaluate scripts: `test.py` and `parallel_test.py`. The `test.py` script is used for single GPU evaluation, while the `parallel_test.py` script is used for multi-GPU evaluation when using one GPU leading to OOM. There are 2 must-have arguments: `model_path`, and `zip_name`. The `model_path` is the path of the checkpoint file, and `zip_name` is the name of the zip file. You can evaluate the model by running

```bash
python test.py --model_path result/exp1/best_model.pth \
               --zip_name best_model \
               --model_type fpnv2 \
               --score_threshold 0.5
```

The `parallel_test.py` script shares the same arguments as `test.py`.

## Performance Snapshot

![image](https://github.com/user-attachments/assets/4400435b-d3e7-4319-a752-ae9a4033ad89)
