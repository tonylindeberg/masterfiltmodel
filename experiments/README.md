# ConvNeXtV2 Training for Modified and Modeled Depthwise Filters

This repo includes two training scripts for ConvNeXtV2 after replacing depthwise convolution filters with mathematically modeled filters.

- `main_finetune.py` fine-tunes a modified ConvNeXtV2 model while **freezing depthwise convolution filters**. (See `modify_models`for how to modify a model's depthwise filters to Madter Key Filters)
- `main_finetune_scale_learn.py` fine-tunes a modeled ConvNeXtV2 model while **learning the scale parameters** of the mathematical depthwise filters.

Both scripts are intended for ImageNet fine-tuning from FCMAE-pretrained checkpoints.

## Training modes

### 1) Fine-tune with frozen depthwise filters

Use this when the depthwise filters have already been replaced and should remain fixed during training.

```bash
torchrun --nproc_per_node=8 main_finetune.py \
  --model convnextv2_tiny \
  --batch_size 256 \
  --update_freq 4 \
  --seed 0 \
  --blr 8e-4 \
  --epochs 300 \
  --warmup_epochs 40 \
  --layer_decay_type single \
  --layer_decay 0.9 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --model_ema True \
  --model_ema_eval True \
  --use_amp True \
  --freeze_dwconv True \
  --finetune ~/pretrained/convnextv2_tiny_1k_224_fcmae_B.pt \
  --data_path ~/data/ILSVRC2012/ \
  --output_dir ~/outputs/FilterSharing/trained_tiny_B
```

### 2) Fine-tune while learning filter scales

Use this when the depthwise filters are mathematically constrained and you want to learn their scale parameters during fine-tuning.

```bash
torchrun --nproc_per_node=8 main_finetune_scale_learn.py \
  --model convnextv2_tiny \
  --batch_size 256 \
  --update_freq 4 \
  --seed 0 \
  --blr 8e-4 \
  --epochs 300 \
  --warmup_epochs 40 \
  --layer_decay_type single \
  --layer_decay 0.9 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --model_ema True \
  --model_ema_eval True \
  --use_amp True \
  --freeze_dwconv False \
  --finetune pretrained/convnextv2_tiny_1k_224_fcmae.pt \
  --data_path ~/data/ILSVRC2012/ \
  --output_dir ~/outputs/FilterSharing/train_tiny_learnscale
```

## Key arguments

- `--model`: ConvNeXtV2 variant to train
- `--finetune`: checkpoint to initialize from
- `--data_path`: path to ImageNet / ILSVRC2012
- `--output_dir`: directory for checkpoints and logs
- `--use_amp`: enables mixed precision training
- `--model_ema` / `--model_ema_eval`: enables EMA training and evaluation
- `--freeze_dwconv`: controls whether depthwise filters are frozen

## Output

Both scripts save checkpoints and training logs to `--output_dir`.
