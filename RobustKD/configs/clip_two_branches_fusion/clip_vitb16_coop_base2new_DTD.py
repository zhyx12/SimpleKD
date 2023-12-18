_base_ = [
    '../_base_/cls_datasets/base_to_new/DTD_add_single_test.py',
    # "../_base_/cls_datasets/base_to_new/DTD_random_resized_crop.py",
    "../_base_/cls_models/clip/clip_vitb16_LoRA_adamw_5e4.py",
]

training_branch_index = (0, 1)

models = dict(
    base_model=dict(
        lora_r=2,
        lora_alpha=2.0,
        text_lora_rank=4,
        text_lora_alpha=4.0,
        optimize_parameters=['lora_a', 'lora_b'],
        # optimize_parameters=['lora_a', 'lora_b', 'extra_prototypes'],
        test_lora_index=105,
        text_test_lora_index=105,
        image_test_merge_index=training_branch_index,
        text_test_merge_index=training_branch_index,
        diversity_loss_index=(0, 1),
        template="{} texture.",
    ),
)

log_interval = 100
val_interval = 400

control = dict(
    log_interval=log_interval,
    max_iters=10000,
    val_interval=val_interval,
    # cudnn_deterministic=True,
    save_interval=500,
    max_save_num=0,
    # seed=480214840,
    # seed=258242457,  # 24011
    # seed=1492777793,  # 97051
    # seed=494886941,
    # seed=333016457, # 73043
    # seed=324670923,
    # seed=324670923, # ce baseline seed
    # seed=1829175462,
    # test_mode=True,
)

train = dict(
    lambda_label_smooth=0.0,
    base_class_list=list(range(24)),
    moving_average_type='simple',
    lambda_kl_loss=1.0,
    lambda_dkd_loss=1.0,
    dkd_loss_alpha=0.0,
    dkd_loss_beta=2.0,
    start_iteration=200,
    num_branches=len(training_branch_index),
    lambda_text_to_text_dkd=0.0,
    add_novel_class_in_dkd=False,
    lambda_div=0.0,
    kl_target_temp=0.01,
    target_type='dynamic',
    loss_type='instance_relation+dkd',
    instance_relation_input_type='feat',
    instance_relation_dkd_temp=0.14,
    use_rand_mask_in_instance_relation=False,
    rand_mask_ratio=0.15,
    detach_text_in_instance_relation=False,
    lambda_instance_relation=0.2,
    fusion_type_in_instance_relation=None,
)

group_acc = {'BASE': list(range(24)),
             'NEW': list(range(24, 47))}

test = dict(
    custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred', class_acc=True, num_class=24),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred', class_acc=True, num_class=23),
        dict(type='ClsAccuracy', dataset_index=2, pred_key='pred', class_acc=True,group_acc=group_acc),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='branch_0_logits', class_acc=True, num_class=24),
        # dict(type='ClsAccuracy', dataset_index=1, pred_key='branch_0_logits', class_acc=True, num_class=23),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='branch_2_logits', class_acc=True, num_class=24),
        # dict(type='ClsAccuracy', dataset_index=1, pred_key='branch_2_logits', class_acc=True, num_class=23),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='mean_logits', class_acc=True, num_class=24),
        # dict(type='ClsAccuracy', dataset_index=1, pred_key='mean_logits', class_acc=True, num_class=23),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='maximum_logits', class_acc=True, num_class=24),
        # dict(type='ClsAccuracy', dataset_index=1, pred_key='maximum_logits', class_acc=True, num_class=23),
        dict(type='ClassRelationshipVis', dataset_index=0, ),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST", pred_key='pred'),
        dict(type='HarmonicMean', pred_key='pred', priority="LOWEST"),
    ]
)
