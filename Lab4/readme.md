# NYCU Computer Vision 2025 Spring HW4

StudentID: 111550089

Name: 李宗諺

## Introduction

In this assignment, we explored the PromptIR architecture to tackle image restoration. PromptIR solves blind image restoration, utilizing learnable prompts to guide the restoration process for various degradation types. I followed the PromptIR pipeline and explored modifications to enhance its performance for dual-degradation image restoration tasks involving both rain and snow. This included varying the prompt length within the PromptGenBlock modules to investigate its impact on restoration quality for specific degradation types. I also enhanced the loss function by combining L1, PSNR, SSIM, and VGG perceptual losses, and implemented model ensemble techniques to boost the eventual performance. The model was trained to restore images degraded by rain and snow, and a PSNR score of 32.11 on the public test dataset was achieved.

## How to install

It is recommended to use `conda` to manage the environment. The following command will create a new environment and install the required packages.

```bash
conda env create -f environment.yml
```

Then, activate the environment by running

```bash
conda activate dlcv_lab4
```

## How to train

You should place the dataset in the `dataset` folder before training. The dataset should be organized as follows:
- `dataset`
  - `test`
    - `degraded`
      - `{i}.png`
  - `train`
    - `clean`
      - `{type}_clean-{i}.png`
    - `degraded`
      - `{type}-{i}.png`

And the main training script is `train.py`. You can run the training script with the following command, where the arguments are explained below:
- `data_root`: the root directory of the dataset.
- `batch_size`: the batch size for training.
- `lr`: the learning rate for training.
- `epochs`: the number of epochs for training.
- `result_dir`: the directory to save the training results, including the best model checkpoint and training log.

```bash
python train.py --data_root ./dataset \
                --batch_size 4 \
                --lr 2e-4 \
                --epochs 150 \
                --result_dir ./results/exp
```

Additionally, you can use the `train2.py` script to train the model in with advanced loss functions (instead of the default L1 loss). The basic arguments are the same as `train.py`, while the additional arguments are explained below:
- `l1_weight`: the weight for L1 loss.
- `psnr_weight`: the weight for PSNR loss.
- `ssim_weight`: the weight for SSIM loss.
- `perceptual_weight`: the weight for VGG perceptual loss.

```bash
python train2.py --data_root ./dataset \
                 --batch_size 4 \
                 --lr 2e-4 \
                 --epochs 150 \
                 --result_dir ./results/exp1 \
                 --l1_weight 1.0 \
                 --psnr_weight 0.1 \
                 --ssim_weight 0.1 \
                 --perceptual_weight 0.01
```

Both `train.py` and `train2.py` store the training results in the specified `result_dir`, including the best model checkpoint and training log. The training logs can be visualized using TensorBoard by running the following command:

```bash
tensorboard --logdir ./results/exp1/logs
```

## How to evaluate

The evaluation script is `test.py`. You can run the evaluation script with the following command, where the arguments are explained below:
- `data_root`: the root directory of the dataset.
- `checkpoint`: the path to the model checkpoint to be evaluated.
- `result_dir`: the directory to save the evaluation results.

```bash
python test.py --data_root ./dataset \
               --checkpoint ./results/exp1/checkpoints/{checkpoint}.ckpt \
               --result_dir ./results/exp1
```

A zip file containing the evaluation results will be saved in the specified `result_dir`.

An option to boost the performance is to use the `ensemble.py` script, which combines the resulting predictions from multiple checkpoints. You can run the ensemble script with the following command, where the arguments are explained below:
- Prediction files: the zip files containing the predictions from different checkpoints.
- `output_dir`: the directory to save the ensemble results.
- `output_name`: the name of the output file, which will be saved as `{output_dir}/{output_name}.zip`.
- `weights`: the weights for each prediction file, which should be a comma-separated list. The weights will be normalized to sum to 1.

```bash
python ensemble.py \
       {./results/exp1/pred.zip} \
       {./results/exp2/pred.zip} \
       {./results/exp3/pred.zip} \
       --output_dir ./results/ensemble
       --output_name ensemble_result \
       --weights 0.4,0.3,0.3
```

## Performance Snapshot

![image](https://github.com/user-attachments/assets/ccd37d3e-1695-482d-86f2-0f1f20a5f074)
