# CVPR2024
This is the code for our paper: Robust Knowledge Distillation for Generalizable Vision-Language Models.

## Project structure

The project structure is presented as follows

```
├──configs
| ├──_base\_
| ├────cls_datasets
| ├────cls_models
| ├──clip_two_branches_fusion
├──data
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

**data**: contain dataset images and labels

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

