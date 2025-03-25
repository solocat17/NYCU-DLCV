# NYCU Computer Vision 2025 Spring HW1

StudentID: 111550089
Name: 李宗諺

## Introduction

## How to install

It is recommended to use `conda` to manage the environment. The following command will create a new environment and install the required packages.

```bash
conda env create -f environment.yml
```

Then, activate the environment by running

```bash
conda activate dlcv_lab1
```

## How to train

There are 2 training python files, both of them support parallel training on multiple GPUs when ensemble training is enabled. You can also specify the id of models to train in the ensemble training for both files.
- The file `ensemble_resnext_101_train.py` can be used to train single or 5 ResNeXt101 models (for ensemble). 2 kind of model design can be chosen, ResNeXt101_32X8D or ResNeXt101_64X4D. You can specify the model to use by modifying the config file.
- The file `ensemble_se_sk_cm_resnext_train.py` can be used to train single or 5 (SE-ResNeXt / SK-ResNeXt) models (for ensemble). 2 kind of model design can be chosen, SE-ResNeXt26d_32x4d or SK-ResNeXt50_32x4d. You can specify the model to use by modifying the config file.

```bash
# training one se_resnext model
python ensemble_se_sk_cm_resnext_train.py --config ensemble_se_sk_cm_resnext_config.txt --models_to_train 0

# training the first 3 out of 5 se_resnext models
python ensemble_se_sk_cm_resnext_train.py --config ensemble_se_sk_cm_resnext_config.txt --models_to_train 0,1,2

# training the 4th, 5th resnext101 model
python ensemble_resnext_101_train.py --config ensemble_resnext_101_config.txt --models_to_train 4,5
```

The training result will be saved in the `saved_moels` folder, along with the weights of the model.

## How to evaluate

Alike the training process, there are 2 evaluation python files. You have to specify the path of the model weights and the config file. Also, the strategy of ensemble training can be specified. Options are 'weighted_avg', 'avg', 'max_conf', 'voting'.
- weighted_avg: default, the final prediction is the weighted average of the predictions of each model.
- avg: the final prediction is the average of the predictions of each model.
- max_conf: the final prediction is the prediction of the model with the highest confidence.
- voting: the final prediction is the prediction with the most votes.

```bash
# evaluate one resnext101 model
python ensemble_resnext_101_test.py --config ensemble_resnext_101_config.txt \
  --model_path saved_models/ensemble/resnext101/best_model_0.pth

# evaluating the ensemble of 5 se_resnext + 5 sk_resnext models, with strategy 'voting'
python ensemble_se_sk_cm_resnext_test.py --config ensemble_se_sk_cm_resnext_config.txt \
  --model_paths saved_models/ensemble/se_sk_resnext/best_model_seresnext_0.pth \
  saved_models/ensemble/se_sk_resnext/best_model_seresnext_1.pth \
  saved_models/ensemble/se_sk_resnext/best_model_seresnext_2.pth \
  saved_models/ensemble/se_sk_resnext/best_model_seresnext_3.pth \
  saved_models/ensemble/se_sk_resnext/best_model_seresnext_4.pth \
  saved_models/ensemble/se_sk_resnext/best_model_skresnext_0.pth \
  saved_models/ensemble/se_sk_resnext/best_model_skresnext_1.pth \
  saved_models/ensemble/se_sk_resnext/best_model_skresnext_2.pth \
  saved_models/ensemble/se_sk_resnext/best_model_skresnext_3.pth \
  saved_models/ensemble/se_sk_resnext/best_model_skresnext_4.pth \
  --strategy voting
```

## Performance Snapshot