# Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance

[Bahjat Kawar](https://bahjat-kawar.github.io), [Roy Ganz](https://www.linkedin.com/in/roy-ganz-270592/), and [Michael Elad](https://elad.cs.technion.ac.il/), Technion.

This is the official code repo for the TMLR paper "Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance".

## Pre-trained models

Coming soon...

## Code instructions

### ImageNet robust classifier train
```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --attack_eps {EPSILON} --attack_steps {ATTACK_STEPS}"
python classifier_rob_train.py --data_dir {IN_PATH} $TRAIN_FLAGS $CLASSIFIER_FLAGS --log_dir {LOG_DIR}
```

where:
- `{IN_PATH}` is the path to ImageNet data
- `{LOG_DIR}` is the directory to log outputs into
- `{EPSILON}` is the epsilon for the adversarial attack
- `{ATTACK_STEPS}` is the number of steps for the adversarial attack


### ImageNet sample
```
python classifier_sample.py --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --classifier_scale {SCALE} --classifier_path {CKPT} --model_path 128x128_diffusion.pt --batch_size {BATCH} --num_samples {SAMPLES} --class_idx_begin {CLASS_BEGIN} --timestep_respacing 250 --seed {SEED}
```

where:
- `{SCALE}` is the classifier scale (best results at `1.0`)
- `{CKPT}` is the classifier checkpoint
- `{BATCH}` is the batch size for generation
- `{SAMPLES}` is the number of samples to generate
- `{SEED}` is the random seed to use
- `{CLASS_BEGIN}` is the ImageNet class index to begin generating from (one class per batch). Useful for generating on multiple GPUs.

### CIFAR-10 robust classifier train
```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 32 --classifier_attention_resolutions 16,8 --classifier_depth 2 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --attack_eps {EPSILON} --attack_steps {ATTACK_STEPS} --attack_type {ATTACK_TYPE}"
python cifar_rob_train.py --data_dir {CIFAR_PATH} $TRAIN_FLAGS $CLASSIFIER_FLAGS --log_dir {LOG_DIR}
```
where:
- `{CIFAR_PATH}` is the path to CIFAR-10 training data
- `{LOG_DIR}` is the directory to log outputs into
- `{EPSILON}` is the epsilon for the adversarial attack
- `{ATTACK_STEPS}` is the number of steps for the adversarial attack
- `{ATTACK_TYPE}` is the threat model for the adversarial attack (`l2` or `linf`)

### CIFAR-10 sample
```
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.1 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
python cifar_classifier_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS --classifier_scale {SCALE} --classifier_path {CKPT} --model_path cifar_diffusion_200k.pt --batch_size {BATCH} --num_samples {SAMPLES} --timestep_respacing 250 --seed {SEED}
```

where:
- `{SCALE}` is the classifier scale
- `{CKPT}` is the classifier checkpoint
- `{BATCH}` is the batch size for generation
- `{SAMPLES}` is the number of samples to generate
- `{SEED}` is the random seed to use


## References and Acknowledgements
This repo is heavily based on:
- [guided-diffusion](https://github.com/openai/guided-diffusion) by OpenAI
- [improved-diffusion](https://github.com/openai/improved-diffusion) by OpenAI

If you find our work useful, please cite:
```
@article{kawar2023enhancing,
    title={Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance},
    author={Bahjat Kawar and Roy Ganz and Michael Elad},
    journal={Transactions on Machine Learning Research},
    year={2023}
}
```