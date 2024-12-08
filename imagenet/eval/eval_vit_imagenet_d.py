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
from imagenet.loader_for_vit import (
    get_corrupted_loader_256,
)
from tqdm import tqdm
from utils import ece
import numpy as np


def test_model(model_fn, params, dataloader, ece_bins, K):
    acc = jnp.zeros(K)
    devices = jax.local_devices()
    n_devices = len(devices)
    total_len = 0

    @partial(jax.pmap)
    def eval_batch(bx, by):
        logits = model_fn(params, bx)
        probs = jax.nn.softmax(logits, axis=-1)
        topK = jax.lax.top_k(probs, k=K)[1]
        return topK

    for batch in tqdm(dataloader):
        bx = jnp.array(batch["image"]._numpy())
        original_by = by = jnp.array(batch["label"]._numpy()) + 1
        batch_len = bx.shape[0]
        total_len += batch_len
        if batch_len % n_devices != 0:
            padded_amount = n_devices - (batch_len % n_devices)
            bx = jnp.concatenate(
                [bx, jnp.zeros((padded_amount, *bx[0].shape), bx.dtype)], axis=0
            )
            by = jnp.concatenate(
                [by, jnp.zeros((padded_amount, *by[0].shape), by.dtype)], axis=0
            )
        bx = jax.device_put_sharded(jnp.split(bx, n_devices, axis=0), devices)
        by = jax.device_put_sharded(jnp.split(by, n_devices, axis=0), devices)
        topK = eval_batch(bx, by)
        topK = jnp.concatenate(topK, axis=0)[:batch_len]
        acc += (topK[:, :, None] == original_by[:, None, :]).sum(axis=(0, 2))
    acc /= total_len
    acc = jnp.cumsum(acc)
    result = {
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
    config = open_json(os.path.join(args.root, "config.json"))
    apply_fn = partial(
        getattr(models, config["model_name"])(
            num_classes=config["num_classes"],
        ).apply,
    )

    def model_fn(params, bx):
        logits = apply_fn({"params": params}, bx, is_training=False)
        return logits

    for split in ["background", "material", "texture"]:
        dataloader = get_corrupted_loader_256(
            [os.path.join("data", f"ImageNet-D/{split}.tfrecord")],
            batch_size=args.batch_size,
            num_label_per_example=3,
        )
        result = test_model(model_fn, params, dataloader, args.ece_bins, args.topK)
        os.makedirs(
            os.path.join(args.root, config["dataset"], "ImageNet-D"), exist_ok=True
        )
        with open(
            os.path.join(args.root, config["dataset"], "ImageNet-D", f"{split}.json"),
            "w",
        ) as out:
            json.dump(result, out)
