import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

import torch
from torch import nn

class AttackerStep:
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        raise NotImplementedError

    def step(self, x, g):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError

    def to_image(self, x):
        return x


# L2 threat model
class L2Step(AttackerStep):
    def project(self, x):
        if self.orig_input is None: self.orig_input = x.detach()
        self.orig_input = self.orig_input.cuda()
        diff = x - self.orig_input
        orig_shape = diff.shape
        diff = diff.reshape(orig_shape[0], -1)
        diff_norm = diff.norm(p=2, dim=1).unsqueeze(0).reshape(diff.shape[0], 1).repeat(1, diff.shape[1])
        eps_repeated = self.eps.unsqueeze(0).reshape(diff.shape[0], 1).repeat(1, diff.shape[1])
        diff_norm = torch.clamp(diff_norm, min=eps_repeated, max=None)
        diff = diff * eps_repeated / (diff_norm + 1e-10)
        diff = diff.reshape(orig_shape)
        assert (diff.reshape(orig_shape[0], -1).norm(p=2, dim=1) <= self.eps + 1e-5).all(), "Eps: " + str(self.eps) + "\n Diff norm: " + str(diff.reshape(orig_shape[0], -1).norm(p=2, dim=1)) + "\n <=: " + str(diff.reshape(orig_shape[0], -1).norm(p=2, dim=1) <= self.eps)
        return self.orig_input + diff

    def step(self, x, g):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + self.step_size.reshape(-1, 1, 1, 1) * scaled_g


def untargeted_pgd_l2(model, X, y, num_iter=7, eps=3, step_size=0.5, timesteps=None, diffusion=None):
    steper = L2Step(eps=eps, orig_input=X.clone().detach(), step_size=step_size)
    for t in range(num_iter):
        X = X.clone().detach().requires_grad_(True).cuda()
        pred = model(X, timesteps=timesteps)
        mask = pred.argmax(-1).eq(y) #stop attacking when misclassified
        if (mask == False).all(): return X.detach() #stop loop when all samples are misclassified
        loss = nn.CrossEntropyLoss(reduction='none')(pred, y)
        loss = torch.mean(loss)
        grad, = torch.autograd.grad(loss, [X])
        X = X.requires_grad_(False)
        X[mask] = steper.step(X, grad)[mask]
        X[mask] = steper.project(X)[mask]
    return X.detach()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir = args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(is_cifar = False,
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            
            logger.log("x")
            
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        cifar_name_style=False,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        cifar_name_style=False,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            #attack the images in the microbatch
            sqrt_alphas = th.from_numpy(diffusion.sqrt_alphas_cumprod).to(device=sub_t.device)[sub_t].float()
            step_size = th.ones_like(sqrt_alphas) * args.attack_eps * 0.5/3
            eps = th.ones_like(sqrt_alphas) * args.attack_eps
            sub_batch = untargeted_pgd_l2(model, sub_batch, sub_labels, num_iter=args.attack_steps, eps=eps, step_size=step_size, timesteps=sub_t)
            #continue regularly
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="log_dir",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        attack_eps = 0.5,
        attack_steps = 7,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
