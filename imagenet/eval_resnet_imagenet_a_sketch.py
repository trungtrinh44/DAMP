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
from imagenet.loader_for_resnet import (
    get_corrupted_loader_256,
)
from tqdm import tqdm
from utils import ece


def test_model(model_fn, params, state, dataloader, ece_bins, K):
    nll = 0
    acc = jnp.zeros(K)
    y_prob = []
    y_true = []
    devices = jax.local_devices()
    n_devices = len(devices)
    total_len = 0

    @partial(jax.pmap)
    def eval_batch(bx, by):
        logits = model_fn(params, state, bx)
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
    args = parser.parse_args()

    checkpointer = Checkpointer(os.path.join(args.root, "checkpoint.pkl"))
    checkpoint = checkpointer.load()
    params = checkpoint["params"]
    state = checkpoint["state"]
    config = open_json(os.path.join(args.root, "config.json"))
    apply_fn = partial(
        getattr(models, config["model"])(
            num_classes=config["num_classes"], bn_axis_name=None, low_res=False
        ).apply,
        mutable=list(state.keys()),
    )

    def model_fn(params, state, bx):
        (logits, _), _ = apply_fn({"params": params, **state}, bx, False)
        return logits

    for split in ["A", "Sketch"]:
        dataloader = get_corrupted_loader_256(
            os.path.join("data", f"ImageNet-{split}.tfrecord"),
            batch_size=args.batch_size,
        )
        result = test_model(
            model_fn, params, state, dataloader, args.ece_bins, args.topK
        )
        os.makedirs(os.path.join(args.root, config["dataset"], split), exist_ok=True)
        with open(
            os.path.join(args.root, config["dataset"], split, f"result.json"),
            "w",
        ) as out:
            json.dump(result, out)
