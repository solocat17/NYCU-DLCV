# NYCU Computer Vision 2025 Spring HW2

StudentID: 111550089

Name: 李宗諺

## Introduction

In this assignment, we explored Faster R-CNN architectures to tackle a digit detection problem. Faster R-CNN is a 2-stage object detection model that consists of 3 components. In this assignment, I focused on altering the backbone network to improve the performance of the model by substituting the backbone network with ResNet variants and MobileNet. Eventually, the overall performance on public dataset goes to 0.36 mAP and 0.75 accuracy.

## How to install

It is recommended to use `conda` to manage the environment. The following command will create a new environment and install the required packages.

```bash
conda env create -f environment.yml
```

Then, activate the environment by running

```bash
conda activate dlcv_lab2
```

## How to train

You should place the model config file in the `configs` folder. The training script supports 6 arguments:
- `--config`: the path of the config file. The default is `configs/default.yml`.
- `--exp_name`: the name of the experiment. The default is `exp1`. The result will be saved in `results/exp1`.
- `--data_dir`: the path of the dataset. The default is `.`, which refers to `./nycu-hw2-data`
- `--wandb`: whether to use wandb for logging. The default is `False`.
- `--wandb_project`: the name of the wandb project. The default is `dlcv_lab2`.
- `--wandb_entity`: the name of the wandb entity.

```bash
python train.py --config configs/default.yml --exp_name exp
```

The training result will be saved in the `results/${exp_name}/` folder, along with the weights of the model.

## How to evaluate

The evaluation script supports 4 arguments:
- `--config`: the path of the config file. The default is `configs/default.yml`.
- `--exp_name`: the name of the experiment. The default is `exp1`. The result will be saved in `results/exp1`.
- `--data_dir`: the path of the dataset. The default is `.`, which refers to `./nycu-hw2-data`
- `--ckpt_path`: the path of the checkpoint file.

```bash
python test.py --config configs/default.yml --exp_name exp --data_dir . --ckpt_path results/${exp_name}/checkpoints/best_model.pth
```

## Performance Snapshot

![image](https://github.com/user-attachments/assets/9784b9e6-80a4-4ab5-95ce-a811597d0f3f)
