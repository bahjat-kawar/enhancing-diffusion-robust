import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    #NUM_CLASSES, #import this from improved_diffusion to accomodate CIFAR-10
    model_and_diffusion_defaults as gd_model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion as gd_create_model_and_diffusion,
    create_gaussian_diffusion as gd_create_gaussian_diffusion,
    create_classifier,
    add_dict_to_argparser as gd_add_dict_to_argparser,
    args_to_dict as gd_args_to_dict,
)

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args, gd_args = create_argparser()
    args, gd_args = args.parse_args(), gd_args.parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion = gd_create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu", logger=logger)
    )
    model.to(dist_util.dev())
    if False: #args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(is_cifar = True, **gd_args_to_dict(gd_args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(gd_args.classifier_path, map_location="cpu", logger=logger)
    )
    classifier.to(dist_util.dev())
    if gd_args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * gd_args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    #sampling
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    i = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        
        # classes for the current batch
        classes = th.tensor([0,1,2,3,4,5,6,7,8,9] * (args.batch_size // 10), device=dist_util.dev())
        
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_model{args.model_path}_ckpt{args.classifier_path}_seed{args.seed}_scale{args.classifier_scale}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=50,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=0.125,
        seed = 0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    gd_defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=50,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=0.125,
        seed = 0,
    )
    gd_defaults.update(gd_model_and_diffusion_defaults())
    gd_defaults.update(classifier_defaults())
    gd_defaults.update(dict(
        attention_resolutions="32,16,8",
        num_heads=4,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        image_size=32,
        classifier_attention_resolutions="16,8",
        classifier_depth=2,
        classifier_width=32,
        classifier_pool="attention",
        classifier_resblock_updown=True,
        classifier_use_scale_shift_norm=True,
    )) #from cmd
    gd_parser = argparse.ArgumentParser()
    gd_add_dict_to_argparser(gd_parser, gd_defaults)
    return parser, gd_parser

if __name__ == "__main__":
    main()
