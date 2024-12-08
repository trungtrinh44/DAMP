import torch

torch.cuda.is_available = lambda: False
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import json
import os
import argparse
from functools import partial
import jax
import jax.numpy as jnp
from checkpointer import Checkpointer
import models
from imagenet.loader_for_vit import get_eval_loader, MEAN_RGB, STDDEV_RGB
from tqdm import tqdm
from utils import ece
import numpy as np

mean_rgb = jnp.array(MEAN_RGB) / 255.0
std_rgb = jnp.array(STDDEV_RGB) / 255.0


def test_model(model_fn, params, dataloader, ece_bins, K, epsilon):
    nll = 0
    acc = jnp.zeros(K)
    y_prob = []
    y_true = []
    devices = jax.local_devices()
    n_devices = len(devices)
    total_len = 0

    @partial(jax.pmap)
    def eval_batch(bx, by):
        # First step: Generate adversarial example
        def get_logits(bx):
            logits = model_fn(params, bx)
            nll = -(
                jax.nn.log_softmax(logits, axis=-1)
                * jax.nn.one_hot(
                    by, num_classes=logits.shape[-1], axis=-1, dtype=jnp.float32
                )
            ).sum(-1)
            return nll.sum()

        bx_grads = jax.grad(get_logits, 0)((bx - mean_rgb) / std_rgb)
        adversarial_bx = jnp.clip(bx + jnp.sign(bx_grads) * epsilon, 0.0, 1.0)
        logits = model_fn(params, (adversarial_bx - mean_rgb) / std_rgb)
        nll = -(
            jax.nn.log_softmax(logits, axis=-1)
            * jax.nn.one_hot(
                by, num_classes=logits.shape[-1], axis=-1, dtype=jnp.float32
            )
        ).sum(-1)
        probs = jax.nn.softmax(logits, axis=-1)
        topK = jax.lax.top_k(probs, k=K)[1]
        return nll, probs, topK

    for batch in tqdm(dataloader):
        bx = jnp.array(batch["image"]._numpy())
        original_by = by = jnp.array(batch["label"]._numpy())
        y_true.append(by)
        batch_len = bx.shape[0]
        total_len += batch_len
        if batch_len % n_devices != 0:
            padded_amount = n_devices - (batch_len % n_devices)
            bx = jnp.concatenate(
                [bx, jnp.zeros((padded_amount, *bx[0].shape), bx.dtype)], axis=0
            )
            by = jnp.concatenate([by, jnp.zeros((padded_amount,), by.dtype)], axis=0)
        bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
        by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
        bnll, probs, topK = eval_batch(bx, by)
        probs = jnp.concatenate(probs, axis=0)[:batch_len]
        nll += jnp.concatenate(bnll, axis=0)[:batch_len].sum()
        topK = jnp.concatenate(topK, axis=0)[:batch_len]
        acc += (topK == original_by[..., None]).sum(axis=0)
        y_prob.append(probs)
    nll /= total_len
    acc /= total_len
    acc = jnp.cumsum(acc)
    y_prob = jnp.concatenate(y_prob, axis=0)
    y_true = jnp.concatenate(y_true, axis=0)
    ece_val = ece(y_prob, y_true, ece_bins)
    result = {
        "nll": float(nll),
        "ece": float(ece_val),
        **{f"top-{k}": float(a) for k, a in enumerate(acc, 1)},
    }
    return result


def open_json(path):
    with open(path) as inp:
        obj = json.load(inp)
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--topK", type=int, default=5)
    parser.add_argument("--epsilons", nargs="+", type=float)
    args = parser.parse_args()

    checkpointer = Checkpointer(os.path.join(args.root, "checkpoint.pkl"))
    checkpoint = checkpointer.load()
    params = checkpoint["params"]
    config = open_json(os.path.join(args.root, "config.json"))
    apply_fn = partial(
        getattr(models, config["model"])(
            num_classes=config["num_classes"],
        ).apply,
    )

    def model_fn(params, bx):
        logits = apply_fn({"params": params}, bx, is_training=False)
        return logits

    split = f"fgsm"
    dataloader = get_eval_loader(
        root="data/ImageNet",
        dtype=np.float32,
        batch_size=args.batch_size,
        before_norm=True,
    )
    os.makedirs(os.path.join(args.root, config["dataset"], split), exist_ok=True)
    results = {}
    for epsilon in args.epsilons:
        result = test_model(
            model_fn, params, dataloader, args.ece_bins, args.topK, epsilon
        )
        print("Epsilon:", result)
        results[epsilon] = result
    with open(
        os.path.join(args.root, config["dataset"], split, f"result.json"),
        "w",
    ) as out:
        json.dump(results, out)
