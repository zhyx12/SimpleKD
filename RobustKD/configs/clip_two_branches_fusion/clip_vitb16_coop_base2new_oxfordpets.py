_base_ = [
    '../_base_/cls_datasets/base_to_new/OxfordPets.py',
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
        template="a photo of a {}, a type of pets.",
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
    # seed=1265315431,
    # seed=1530860850,
    # test_mode=True,
    # seed=888,
    save_best_model=False,
)

train = dict(
    lambda_label_smooth=0.0,
    base_class_list=list(range(19)),
    moving_average_type='simple',
    lambda_kl_loss=1.0,
    lambda_dkd_loss=1.0,
    dkd_loss_alpha=0.0,
    dkd_loss_beta=2.0,
    start_iteration=200,
    num_branches=len(training_branch_index),
    lambda_text_to_text_dkd=0.0,
    add_novel_class_in_dkd=False,
    lambda_div=0.02,
    kl_target_temp=0.01,
    target_type='orig_instance',
    loss_type='instance_relation+dkd',
    instance_relation_input_type='feat_representation',
    instance_relation_dkd_temp=0.14,
    use_rand_mask_in_instance_relation=False,
    rand_mask_ratio=0.15,
    detach_text_in_instance_relation=False,
    lambda_instance_relation=0.5,
    fusion_type_in_instance_relation=None,
    use_orig_output_in_instance_relation=True,
)

test = dict(
    custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred', class_acc=True, num_class=19),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred', class_acc=True, num_class=18),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST", pred_key='pred'),
        dict(type='HarmonicMean', pred_key='pred', priority="LOWEST"),
    ]
)
