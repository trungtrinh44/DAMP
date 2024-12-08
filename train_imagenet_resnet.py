import sys
from time import time
import flax.jax_utils
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import torch

torch.cuda.is_available = lambda: False
import json
import logging
import os
from functools import partial
import flax
import jax
import jax.numpy as jnp
from checkpointer import Checkpointer
import models
from imagenet.loader_for_resnet import (
    get_training_loader,
    get_eval_loader,
    get_corrupted_loader,
    IMAGE_SIZE,
    TRAINING_SIZE,
    NUM_CLASSES,
    CORRUPTIONS,
)
from optimizers import (
    build_ASAM_optimizer,
    build_SAM_optimizer,
    build_DAMP_optimizer,
    build_standard_optimizer,
    nesterov,
    PMAP_BATCH,
    VMAP_BATCH,
)
from utils import ece
import argparse


def init_model(model_name, num_classes, input_size, key):
    model = getattr(models, model_name)(
        num_classes=num_classes, bn_axis_name=None, low_res=False
    )
    variables = model.init(key, jnp.ones((1, *input_size)), True)
    state, params = flax.core.pop(variables, "params")
    del variables
    return params, state


def get_learning_rate_scheduler(
    num_epochs, num_warmup_epochs, step_per_epoch, init_lr, lr_ratio, batch_size
):
    max_lr = init_lr * (batch_size / 256)
    min_lr = max_lr * lr_ratio
    num_warmup_steps = num_warmup_epochs * step_per_epoch
    num_cosine_steps = (num_epochs - num_warmup_epochs) * step_per_epoch

    def linear_schedule(i):
        return (max_lr - min_lr) * i / num_warmup_steps + min_lr

    def cosine_schedule(i):
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * i / num_cosine_steps))
        decayed = (1 - lr_ratio) * cosine_decay + lr_ratio
        return max_lr * decayed

    linear_lrs = linear_schedule(jnp.arange(num_warmup_steps))
    cosine_lrs = cosine_schedule(jnp.arange(num_cosine_steps))
    all_lrs = jnp.concatenate([linear_lrs, cosine_lrs])
    return lambda i: all_lrs[i]


def get_optimizer(args, mask):
    scheduler = get_learning_rate_scheduler(
        args.num_epochs,
        args.num_warmup_epochs,
        args.step_per_epoch,
        args.initial_lr,
        args.lr_ratio,
        args.train_batch_size,
    )
    base_optim = nesterov(scheduler, args.momentum, args.weight_decay, mask=mask)
    if args.method == "SGD":
        return build_standard_optimizer(base_optim), scheduler
    if args.method == "DAMP":
        return (
            build_DAMP_optimizer(
                args.std, args.num_microbatches_per_batch // args.n_devices, base_optim
            ),
            scheduler,
        )
    if args.method == "SAM":
        return (
            build_SAM_optimizer(
                args.rho, args.num_microbatches_per_batch // args.n_devices, base_optim
            ),
            scheduler,
        )
    if args.method == "ASAM":
        return (
            build_ASAM_optimizer(
                args.rho,
                args.num_microbatches_per_batch // args.n_devices,
                args.eta,
                base_optim,
            ),
            scheduler,
        )


def compare_and_select_first(x):
    for i in range(1, len(x)):
        assert jnp.allclose(x[0], x[i])
    return x[0]


def save_config(root_dir, args):
    with open(os.path.join(root_dir, "config.json"), "w") as out:
        json.dump(args.__dict__, out)


def setup_logger(root_dir):
    logger = logging.getLogger("Training")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(root_dir, "train.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logger


def test_model(logit_fn, dataloader, ece_bins):
    nll = 0
    acc = [0, 0, 0]
    y_prob = []
    y_true = []
    devices = jax.local_devices()
    n_devices = len(devices)
    assert (
        args.num_microbatches_per_batch % n_devices == 0
    ), "Number of microbatches per batch must be divisible by the number of GPUs"
    total_len = 0

    @partial(jax.pmap, axis_name=PMAP_BATCH)
    def eval_batch(bx, by):
        logits = logit_fn(bx)
        nll = (
            -(
                jax.nn.log_softmax(logits, axis=-1)
                * jax.nn.one_hot(
                    by, num_classes=logits.shape[-1], axis=-1, dtype=jnp.float32
                )
            )
            .sum(-1)
            .sum()
        )
        probs = jax.nn.softmax(logits, axis=-1)
        top3 = jax.lax.top_k(probs, k=3)[1]
        return nll, probs, top3

    for batch in dataloader:
        bx = jnp.array(batch["image"]._numpy())
        by = jnp.array(batch["label"]._numpy())
        total_len += bx.shape[0]
        bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
        by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
        bnll, probs, top3 = eval_batch(bx, by)
        probs = jnp.concatenate(probs, axis=0)
        top3 = jnp.concatenate(top3, axis=0)
        by = jnp.concatenate(by, axis=0)
        nll += bnll.sum()
        y_prob.append(probs)
        y_true.append(by)
        for k in range(3):
            acc[k] += (top3[:, k] == by).sum()
    nll /= total_len
    for k in range(3):
        acc[k] /= total_len
    acc = jnp.cumsum(jnp.array(acc))
    y_prob = jnp.concatenate(y_prob, axis=0)
    y_true = jnp.concatenate(y_true, axis=0)
    ece_val = ece(y_prob, y_true, ece_bins)
    result = {
        "nll": float(nll),
        "ece": float(ece_val),
        **{f"top-{k}": float(a) for k, a in enumerate(acc, 1)},
    }
    return result


def main(args):
    tf.random.set_seed(args.seed)
    devices = jax.local_devices()
    n_devices = len(devices)
    assert (
        args.num_microbatches_per_batch % n_devices == 0
    ), "Number of microbatches per batch must be divisible by the number of GPUs"
    root_dir = os.path.join(
        args.base_dir,
        args.method,
        args.model,
        args.experiment_id,
    )
    os.makedirs(root_dir)
    logger = setup_logger(root_dir)
    args.n_devices = n_devices
    save_config(root_dir, args)
    logger.info(f"The experiment files are stored at {root_dir}")
    if args.previous_state == "":
        # Start training from scratch
        @partial(jax.jit, backend="cpu")
        def init_model_and_seed():
            rng = jax.random.PRNGKey(args.seed)
            key, subkey = jax.random.split(rng, 2)
            params, state = init_model(
                args.model, args.num_classes, args.input_size, subkey
            )
            return params, state, key

        params, state, key = init_model_and_seed()
        key, subkey = jax.random.split(key, 2)
        (opt_init, opt_update, get_model_from_state), scheduler = get_optimizer(
            args,
            mask=jax.tree_util.tree_map_with_path(
                lambda n, _: n[-1].key == "kernel", params
            ),  # apply weight decay only to the weight matrices, skipping the batch norm stats and biases,
        )
        single_trainstate = opt_init(params, subkey, state)
        trainstate = flax.jax_utils.replicate(single_trainstate)
        mutable_keys = list(state.keys())
        del single_trainstate
        del params
        del state
        start_from_step = 0
    else:
        # Start training from a previous state
        previous_state = Checkpointer(args.previous_state).load()
        start_from_step = previous_state["step"]
        trainstate = previous_state["state"]
        mutable_keys = list(trainstate.netstate.keys())
        (opt_init, opt_update, get_model_from_state), scheduler = get_optimizer(
            args,
            mask=jax.tree_util.tree_map_with_path(
                lambda n, _: n[-1].key == "kernel", trainstate.params
            ),  # apply weight decay only to the weight matrices, skipping the batch norm stats and biases,
        )
        del previous_state
    start_from_epochs = start_from_step // args.step_per_epoch
    apply_fn = partial(
        getattr(models, args.model)(
            num_classes=args.num_classes,
            bn_axis_name=(PMAP_BATCH, VMAP_BATCH) if args.sync_batch_norm else None,
            low_res=False,
        ).apply,
        mutable=mutable_keys,
    )
    train_loader = get_training_loader(
        root="data/ImageNet",
        dtype=np.float32,
        batch_size=args.train_batch_size,
        repeat=args.num_epochs - start_from_epochs,
        cache=True,
    )

    @partial(jax.pmap, axis_name=PMAP_BATCH, donate_argnums=(0,))
    def train_step(trainstate, bx, by):
        def forward(params, state, batch, is_training):
            bx, labels = batch
            (logits, _), new_state = apply_fn(
                {"params": params, **state}, bx, is_training
            )
            cross_ent_loss = (
                -(jax.nn.log_softmax(logits, axis=-1) * labels).sum(-1).mean()
            )
            return cross_ent_loss, new_state

        lossgrad = jax.value_and_grad(forward, 0, has_aux=True)
        # Create (smoothed) one-hot labels
        labels = jax.nn.one_hot(
            by, num_classes=args.num_classes, axis=-1, dtype=jnp.float32
        )
        if args.label_smoothing > 0.0:
            labels = (
                1.0 - args.label_smoothing
            ) * labels + args.label_smoothing * jnp.ones_like(labels) / args.num_classes
        newtrainstate, loss = opt_update(lossgrad, trainstate, (bx, labels))
        return newtrainstate, loss

    step_count = start_from_step
    total_time = 0
    t0 = time()
    trainstate_checkpointer = Checkpointer(os.path.join(root_dir, "trainstate.pkl"))
    logger.info(
        f"Start from step {start_from_step} and from epochs {start_from_epochs}"
    )
    for batch in train_loader:
        if step_count > args.num_epochs * args.step_per_epoch:
            break
        step_count += 1
        bx = jnp.array(batch["image"]._numpy())
        by = jnp.array(batch["label"]._numpy())
        bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
        by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
        trainstate, loss = train_step(trainstate, bx, by)
        loss = jnp.mean(loss).item()
        step_size = scheduler(step_count)
        t1 = time()
        total_time += t1 - t0
        t0 = t1
        logger.info(
            f"Step {step_count}: neg_log_like {loss:.4f}, lr {step_size:.4f}, time {total_time/(step_count-start_from_step):.4f}"
        )
        if step_count % args.save_freq == 0:
            trainstate_checkpointer.save({"state": trainstate, "step": step_count})
            logger.info(f"Save state at {step_count}")
    checkpointer = Checkpointer(os.path.join(root_dir, "checkpoint.pkl"))
    trainstate_checkpointer.save({"state": trainstate, "step": step_count})
    trained_model = get_model_from_state(trainstate)
    checkpointer.save(
        {
            "params": jax.tree_util.tree_map(
                compare_and_select_first, trained_model["params"]
            ),
            "state": jax.tree_util.tree_map(
                compare_and_select_first if args.sync_batch_norm else lambda s: s[0],
                trained_model["state"],
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--base_dir", default="experiments", type=str)
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--model", default="ResNet50", type=str)
    parser.add_argument("--num_ece_bins", default=15, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=512, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--initial_lr", type=float, default=0.1)
    parser.add_argument("--lr_ratio", default=0.001, type=float)
    parser.add_argument("--num_warmup_epochs", default=0, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_epochs", default=90, type=int)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--num_microbatches_per_batch", default=8, type=int)
    parser.add_argument("--sync_batch_norm", action="store_true")
    parser.add_argument("--previous_state", default="", type=str)
    parser.add_argument("--save_freq", type=int, default=5)
    subparsers = parser.add_subparsers(
        help="Choosing optimization methods", dest="method"
    )
    parser_DAMP = subparsers.add_parser("DAMP")
    parser_DAMP.add_argument("--std", type=float, required=True)

    parser_SAM = subparsers.add_parser("SAM")
    parser_SAM.add_argument("--rho", type=float, required=True)

    parser_ASAM = subparsers.add_parser("ASAM")
    parser_ASAM.add_argument("--rho", type=float, required=True)
    parser_ASAM.add_argument("--eta", type=float, default=0.01)

    parser_SGD = subparsers.add_parser("SGD")

    args = parser.parse_args()
    if args.conf is not None:
        with open(args.conf, "r") as f:
            parser.set_defaults(**json.load(f))

        # Reload arguments to override config file values with command line values
        args = parser.parse_args()
        del args.conf

    args.dataset = "imagenet"
    args.num_classes = NUM_CLASSES
    args.input_size = (IMAGE_SIZE, IMAGE_SIZE, 3)
    args.step_per_epoch = TRAINING_SIZE // args.train_batch_size
    args.save_freq = args.save_freq * args.step_per_epoch
    main(args)
