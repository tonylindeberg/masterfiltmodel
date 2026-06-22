# Master Key Filter Replacement for ConvNeXtV2

This script modifies a ConvNeXtV2 model by replacing its depthwise convolution filters with the best-fitting **Master Key filters**.

The Master Key filters are mathematically derived from the methods introduced in the paper *"Modeling and Analysis of the 8 Filters from the “Master Key Filters
Hypothesis” for Depthwise-Separable Deep Networks in Relation to Idealized Receptive Fields Based on Scale-Space Theory"*, with **Method B** used by default. The script compares filters from a **trained model** against the predefined Master Key filters, then applies the best-fit estimates to an **FCMAE-pretrained ConvNeXtV2 model**. Pretrained checkpoints can be directly downloaded from the official [ConvNeXt V2 repository](https://github.com/facebookresearch/ConvNeXt-V2).

## What it does

- Loads a trained ConvNeXtV2 model
- Generates Master Key filters using a selected method (`A`, `B`, `C1`, `C2`, `D1`, or `D2`)
- Finds the best-fitting filter approximation for each depthwise convolution filter
- Replaces the depthwise convolution filters in the FCMAE-pretrained model
- Saves the modified model checkpoint

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- SciPy


## Usage

```bash
python modify_dwconvs_initial.py \
  --model_type convnextv2 \
  --trained_model_path /path/to/trained_model.pt \
  --model_path /path/to/fcmae_model.pt \
  --modified_model_path /path/to/output_model.pt \
  --method B
```

## Arguments

- `--model_type`  
  Model type. Default: `convnextv2`

- `--trained_model_path`  
  Path to the trained model checkpoint used to find best-fit filters

- `--model_path`  
  Path to the FCMAE-pretrained model checkpoint to be modified

- `--modified_model_path`  
  Path where the modified model will be saved

- `--method`  
  Master Key filter generation method. Options: `A`, `B`, `C1`, `C2`, `D1`, `D2`  
  Default: `B`

## Example

```bash
python modify_dwconvs_initial.py \
  --trained_model_path ./checkpoints/convnextv2_trained.pt \
  --model_path ./checkpoints/convnextv2_fcmae.pt \
  --modified_model_path ./checkpoints/convnextv2_fcmae_methodB.pt \
  --method B
```

## Notes

- This script is designed for **ConvNeXtV2** checkpoints.
- Only `dwconv.weight` layers are modified.
- The output checkpoint keeps the original model structure, with updated depthwise convolution filters.

## Output

The script saves a new model checkpoint at the path specified by `--modified_model_path`.
