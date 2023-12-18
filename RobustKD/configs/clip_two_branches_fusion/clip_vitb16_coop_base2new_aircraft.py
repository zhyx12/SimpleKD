_base_ = [
    '../_base_/cls_datasets/base_to_new/Aircraft.py',
    "../_base_/cls_models/clip/clip_vitb16_LoRA_adamw_5e4.py",
]

training_branch_index = (0, 1)

models = dict(
    base_model=dict(
        lora_r=4,
        lora_alpha=4.0,
        text_lora_rank=4,
        text_lora_alpha=4.0,
        optimize_parameters=['lora_a', 'lora_b'],
        test_lora_index=105,
        text_test_lora_index=105,
        image_test_merge_index=(0, 1),
        text_test_merge_index=(0, 1),
        diversity_loss_index=(0, 1),
        template="a photo of a {}, a type of aircraft.",
    ),
)

log_interval = 100
val_interval = 400

control = dict(
    log_interval=log_interval,
    max_iters=10000,
    val_interval=val_interval,
    save_interval=500,
    max_save_num=0,
    save_best_model=False,
)

train = dict(
    base_class_list=list(range(50)),
    moving_average_type='simple',
    num_branches=len(training_branch_index),
    target_type='orig_instance',
    instance_relation_input_type='feat_representation',
)

test = dict(
    custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred', class_acc=True, num_class=50),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred', class_acc=True, num_class=50),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST", pred_key='pred'),
        dict(type='HarmonicMean', pred_key='pred', priority="LOWEST"),
    ]
)
