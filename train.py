import sys
import time
import flax.jax_utils
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
from datasets import get_corrupt_data_loader, get_data_loader
from optimizers import (
    build_ASAM_optimizer,
    build_SAM_optimizer,
    build_DAMP_optimizer,
    build_standard_optimizer,
    nesterov,
    PMAP_BATCH,
    VMAP_BATCH,
)
from utils import LrScheduler, ece
import argparse
from datetime import datetime


def init_model(model_name, num_classes, input_size, key):
    model = getattr(models, model_name)(num_classes=num_classes, bn_axis_name=None)
    variables = model.init(key, jnp.ones((1, *input_size)), True)
    state, params = flax.core.pop(variables, "params")
    del variables
    return params, state


def get_optimizer(args):
    scheduler = LrScheduler(
        args.initial_lr,
        args.num_epochs,
        args.milestones,
        args.lr_ratio,
        args.num_start_epochs,
        args.step_per_epoch,
    )
    base_optim = nesterov(scheduler, args.momentum, args.weight_decay)
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


def get_dataloader(
    train_batch_size,
    test_batch_size,
    validation,
    validation_fraction,
    dataset,
    augment_data,
    num_train_workers,
    num_test_workers,
    seed,
):
    return get_data_loader(
        dataset,
        train_bs=train_batch_size,
        test_bs=test_batch_size,
        validation=validation,
        validation_fraction=validation_fraction,
        augment=augment_data,
        num_train_workers=num_train_workers,
        num_test_workers=num_test_workers,
        valid_split_seed=seed,
    )


def get_corrupt_dataloader(intensity, test_batch_size, dataset, num_test_workers):
    return get_corrupt_data_loader(
        dataset,
        intensity,
        batch_size=test_batch_size,
        root_dir="data/",
        num_workers=num_test_workers,
    )


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


def test_model(get_logits, dataloader, ece_bins):
    nll = 0
    acc = [0, 0, 0]
    y_prob = []
    y_true = []
    devices = jax.local_devices()
    n_devices = len(devices)
    assert (
        args.num_microbatches_per_batch % n_devices == 0
    ), "Number of microbatches per batch must be divisible by the number of GPUs"

    @partial(jax.pmap, axis_name=PMAP_BATCH)
    def eval_batch(bx, by):
        logits = get_logits(bx)
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

    for bx, by in dataloader:
        bx = jnp.array(bx.permute(0, 2, 3, 1).numpy())
        by = jnp.array(by.numpy())
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
    nll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
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


def compare_and_select_first(x):
    for i in range(1, len(x)):
        assert jnp.allclose(x[0], x[i])
    return x[0]


def save_config(root_dir, args):
    with open(os.path.join(root_dir, "config.json"), "w") as out:
        json.dump(args.__dict__, out)


def main(args):
    devices = jax.local_devices()
    n_devices = len(devices)
    assert (
        args.num_microbatches_per_batch % n_devices == 0
    ), "Number of microbatches per batch must be divisible by the number of GPUs"
    root_dir = os.path.join(
        args.base_dir,
        args.method,
        args.model,
        args.dataset,
        args.experiment_id,
    )
    os.makedirs(root_dir)
    args.n_devices = n_devices
    save_config(root_dir, args)
    dataloaders = get_dataloader(
        args.train_batch_size,
        args.test_batch_size,
        args.validation,
        args.validation_fraction,
        args.dataset,
        True,
        args.num_train_workers,
        args.num_test_workers,
        args.seed,
    )
    logger = setup_logger(root_dir)
    logger.info(f"The experiment files are stored at {root_dir}")
    if args.validation:
        train_loader, valid_loader, test_loader = dataloaders
        logger.info(
            f"Train size: {len(train_loader.dataset)}, validation size: {len(valid_loader.dataset)}, test size: {len(test_loader.dataset)}"
        )
    else:
        train_loader, test_loader = dataloaders
        logger.info(
            f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}"
        )
    args.step_per_epoch = len(train_loader)

    @partial(jax.jit, backend="cpu")
    def init_model_and_seed():
        rng = jax.random.PRNGKey(args.seed)
        key, subkey = jax.random.split(rng, 2)
        params, state = init_model(
            args.model, args.num_classes, args.input_size, subkey
        )
        return params, state, key

    params, state, key = init_model_and_seed()
    if args.sync_batch_norm:
        bn_axis_name = (PMAP_BATCH, VMAP_BATCH)
    else:
        bn_axis_name = None
    apply_fn = partial(
        getattr(models, args.model)(
            num_classes=args.num_classes,
            bn_axis_name=bn_axis_name,
        ).apply,
        mutable=list(state.keys()),
    )
    (opt_init, opt_update, get_model_from_state), scheduler = get_optimizer(args)
    key, subkey = jax.random.split(key, 2)
    single_trainstate = opt_init(params, subkey, state)
    trainstate = flax.jax_utils.replicate(single_trainstate)
    del params
    del state
    del single_trainstate

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
        labels = jax.nn.one_hot(
            by, num_classes=args.num_classes, axis=-1, dtype=jnp.float32
        )
        new_trainstate, loss = opt_update(lossgrad, trainstate, (bx, labels))
        return new_trainstate, loss

    for i in range(args.num_epochs):
        avg_loss = 0.0
        start_time = time.time()
        for j, (bx, by) in enumerate(train_loader):
            bx = jnp.array(bx.permute(0, 2, 3, 1).numpy())
            by = jnp.array(by.numpy())
            bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
            by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
            trainstate, loss = train_step(trainstate, bx, by)
            loss = jnp.mean(loss).item()
            avg_loss += loss
            step_size = scheduler(i * args.step_per_epoch + j)
            logger.info(
                f"Epoch {i}, iteration {j}: loss {loss:.4f}, lr {step_size:.4f}"
            )
        duration = time.time() - start_time
        avg_loss /= len(train_loader)
        logger.info(
            f"Epoch {i}: avg loss {avg_loss:.4f}, lr {step_size:.4f}, duration {duration:.2f}s"
        )

    checkpointer = Checkpointer(os.path.join(root_dir, "checkpoint.pkl"))
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

    checkpoint = checkpointer.load()
    params = checkpoint["params"]
    state = checkpoint["state"]

    def get_logits(bx):
        (logits, _), _ = apply_fn({"params": params, **state}, bx, False)
        return logits

    test_result = test_model(get_logits, test_loader, args.num_ece_bins)
    os.makedirs(os.path.join(root_dir, args.dataset), exist_ok=True)
    with open(os.path.join(root_dir, args.dataset, "test_result.json"), "w") as out:
        json.dump(test_result, out)
    if args.validation:
        valid_result = test_model(get_logits, valid_loader, args.num_ece_bins)
        with open(
            os.path.join(root_dir, args.dataset, "valid_result.json"), "w"
        ) as out:
            json.dump(valid_result, out)
    for i in range(5):
        dataloader = get_corrupt_dataloader(
            i, args.test_batch_size, args.dataset, args.num_test_workers
        )
        result = test_model(get_logits, dataloader, args.num_ece_bins)
        os.makedirs(os.path.join(root_dir, args.dataset, str(i)), exist_ok=True)
        with open(
            os.path.join(root_dir, args.dataset, str(i), "result.json"), "w"
        ) as out:
            json.dump(result, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--base_dir", default="experiments", type=str)
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--model", default="ResNet18", type=str)
    parser.add_argument("--num_ece_bins", default=15, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=512, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["cifar10", "cifar100", "tinyimagenet"],
    )
    parser.add_argument("--initial_lr", type=float, default=0.1)
    parser.add_argument("--lr_ratio", default=0.01, type=float)
    parser.add_argument("--milestones", default=[0.5, 0.9], type=float, nargs=2)
    parser.add_argument("--num_start_epochs", default=0, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--num_train_workers", default=8, type=int)
    parser.add_argument("--num_test_workers", default=2, type=int)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--num_microbatches_per_batch", default=8, type=int)
    parser.add_argument("--sync_batch_norm", action="store_true")
    subparsers = parser.add_subparsers(
        help="Choosing optimization methods", dest="method"
    )
    parser_DAMP = subparsers.add_parser("DAMP")
    parser_DAMP.add_argument("--std", type=float, default=0.0)

    parser_SAM = subparsers.add_parser("SAM")
    parser_SAM.add_argument("--rho", type=float, default=0.0)

    parser_ASAM = subparsers.add_parser("ASAM")
    parser_ASAM.add_argument("--rho", type=float, default=0.0)
    parser_ASAM.add_argument("--eta", type=float, default=0.0)

    parser_SGD = subparsers.add_parser("SGD")

    args = parser.parse_args()
    if args.conf is not None:
        with open(args.conf, "r") as f:
            parser.set_defaults(**json.load(f))

        # Reload arguments to override config file values with command line values
        args = parser.parse_args()
        del args.conf

    if args.dataset == "cifar100":
        args.num_classes = 100
        args.input_size = (32, 32, 3)
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.input_size = (32, 32, 3)
    elif args.dataset == "tinyimagenet":
        args.num_classes = 200
        args.input_size = (64, 64, 3)
    main(args)
