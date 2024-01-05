# CVPR2024
This is the code for our paper: Robust Knowledge Distillation for Generalizable Vision-Language Models.  The code.zip contains everything here and can be downloaded directly.

## Project structure

The project structure is presented as follows

```
├──configs
| ├──_base\_
| ├────cls_datasets
| ├────cls_models
| ├──clip_two_branches_fusion
├──data
| ├──Aircraft
| ├──Caltech101
| ├──txt
| | ├──base2new
| | | ├──train_images_Aircraft.txt
├──runs
├──robustkd
| ├──hooks
| ├──loaders
| ├──models
| ├──trainers
├──experiments
| ├──clip_distill_train.sh
├──train.py
```

**configs**: training configs files for different experiments

**data**: contain dataset images and labels, the txt folder stores files containing the image path of different splits (\e.g., train, val, test) for each dataset

**runs**: automatically created which stores checkpoints, tensorboard and text logging files

**robustkd**: source code of our method, contains hooks (evaluation hooks), loaders (train and test loaders), models (
definition of models), trainers (training and testing process)

**experiments**: training scripts

**train.py**: entrance to training process

# Core files

1. Model definition:

   ./robustkd/models/clip_lora_distill.py

2. Training process:

   ./CRCo/trainers/trainer_clip_distill.py

# Steps to reproduce the results

Here we take Aircraft under the base-to-new task setting as an example.

1. Some preparations.

- Make a folder '/home/username/PycharmProjects', replace the username by your name under /home.
- Download the "code.zip" here. Unzip it under '/home/username/PycharmProjects', and rename this folder by 'RobustKD' (needed in [clip_distill_train.sh](RobustKD/experiments/clip_distill_train.sh))

2. Go to the project home

```
cd /home/username/PycharmProjects/RobustKD
```

3. Train target domain model using the following script.

```
CUDA_VISIBLE_DEVICES=0 bash ./experiments/clip_distill_train.sh my_exp ./configs/clip_two_branches_fusion/clip_vitb16_coop_base2new_aircraft.py
```

The results of all training metrics and testing accuracy is stored in the **runs** folder. You can view them from the txt log file or through tensorboard.
