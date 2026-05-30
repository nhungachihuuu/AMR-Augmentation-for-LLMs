import math


def adjust_learning_rate(param_group, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr


def adjust_lr_step(param_group, update_step, total_steps, base_lr, min_lr, warmup_steps):
    # update_step is 0-based index of optimizer updates
    if update_step < warmup_steps:
        lr = base_lr * float(update_step + 1) / max(1, warmup_steps)
    else:
        progress = float(update_step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    param_group["lr"] = lr
    return lr


