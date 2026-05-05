import part_3_train_config
import part_3_training

# ── Step 1: Prepare config files for all LoRA experiments ──────────────────
part_3_train_config.prepare_configs(device_type='cuda', has_logs=False)

# ── Step 2: Run all experiments ────────────────────────────────────────────

# Experiment 4: LoRA single-task SFT (rank 4)
part_3_training.train('exp4_taskA_rank4')
part_3_training.train('exp4_taskB_rank4')

# Experiment 5: Rank ablation on Task A
part_3_training.train('exp5_taskA_rank1')
part_3_training.train('exp5_taskA_rank2')
# part_3_training.train('exp5_taskA_rank4')
part_3_training.train('exp5_taskA_rank8')
part_3_training.train('exp5_taskA_rank16')

# Experiment 6: LoRA vs full fine-tuning
# part_3_training.train('exp6_fullFT')
# part_3_training.train('exp6_lora_rank4')
part_3_training.train('exp6_lora_multitask')