from functools import partial
from optax import GradientTransformation, apply_updates
import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional
from .constants import PMAP_BATCH, VMAP_BATCH
from .utils import sample_like_tree


class TrainState(NamedTuple):
    """
    collects the all the state required for neural network training
    """

    optstate: dict
    netstate: Optional[dict]
    params: dict
    rngkey: None


def build_standard_optimizer(optimizer: GradientTransformation):
    def init(params, rngkey=None, netstate=None):
        optstate = optimizer.init(params)
        return TrainState(
            optstate=optstate, params=params, netstate=netstate, rngkey=None
        )

    def get_model_from_state(trainstate):
        return {"params": trainstate.params, "state": trainstate.netstate}

    def _gradient(lossgrad, trainstate, X_subbatch, y_subbatch):
        # gradient at current point
        netstate = trainstate.netstate
        if netstate is None:
            loss, grad = lossgrad(
                trainstate.params,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            (loss, netstate), grad = lossgrad(
                trainstate.params,
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        return grad, loss, netstate

    def step(loss_fn, trainstate, minibatch):
        optstate = trainstate.optstate
        X_batch, y_batch = minibatch
        grad, loss, netstate = _gradient(loss_fn, trainstate, X_batch, y_batch)
        grad = jax.lax.pmean(
            grad,
            axis_name=PMAP_BATCH,
        )
        grad, optstate = optimizer.update(grad, optstate, trainstate.params)
        params = apply_updates(trainstate.params, grad)

        newtrainstate = trainstate._replace(
            optstate=optstate, params=params, netstate=netstate
        )

        return newtrainstate, loss

    return init, step, get_model_from_state


def build_DAMP_optimizer(
    noise_std: float,
    batchsplit: float,
    optimizer: GradientTransformation,
):
    def filter_fn(name):
        return name[-1].key in ("kernel", "scale")

    def init(params, rngkey, netstate=None):
        optstate = optimizer.init(params)
        if netstate is not None:
            netstate = jax.tree_util.tree_map(
                lambda s: jnp.repeat(s[None, ...], repeats=batchsplit, axis=0), netstate
            )
        return TrainState(
            optstate=optstate, params=params, netstate=netstate, rngkey=rngkey
        )

    def get_model_from_state(trainstate):
        return {
            "params": trainstate.params,
            "state": jax.tree_util.tree_map(lambda s: s[0], trainstate.netstate),
        }

    def _gradient(lossgrad, params, netstate, X_subbatch, y_subbatch, rngkey):
        # gradient at current point
        if netstate is None:
            loss, grad = lossgrad(
                _build_noisy_params(params, rngkey),
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            (loss, netstate), grad = lossgrad(
                _build_noisy_params(params, rngkey),
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        return grad, loss, netstate

    def _build_noisy_params(params, key):
        noise, _ = sample_like_tree(params, key)
        params = jax.tree_util.tree_map_with_path(
            lambda n, p, z: (p * (1 + z * noise_std) if filter_fn(n) else p),
            params,
            noise,
        )
        return params

    def step(loss_fn, trainstate, minibatch):
        X_batch, y_batch = minibatch
        rngkey = jax.random.fold_in(trainstate.rngkey, jax.lax.axis_index(PMAP_BATCH))
        rngkey = jax.random.split(rngkey, batchsplit + 1)
        X_batch = jnp.reshape(X_batch, (batchsplit, -1, *X_batch.shape[1:]))
        y_batch = jnp.reshape(y_batch, (batchsplit, -1, *y_batch.shape[1:]))
        grad, loss, netstate = jax.vmap(
            partial(_gradient, loss_fn), (None, 0, 0, 0, 0), 0, axis_name=VMAP_BATCH
        )(trainstate.params, trainstate.netstate, X_batch, y_batch, rngkey[1:])
        grad = jax.lax.pmean(
            jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grad),
            axis_name=PMAP_BATCH,
        )
        grad, optstate = optimizer.update(grad, trainstate.optstate, trainstate.params)
        params = apply_updates(trainstate.params, grad)

        newtrainstate = trainstate._replace(
            optstate=optstate, params=params, rngkey=rngkey[0], netstate=netstate
        )

        return newtrainstate, jnp.mean(loss)

    return init, step, get_model_from_state


def build_SAM_optimizer(
    rho: float,
    msharpness: int,
    optimizer: GradientTransformation,
):

    def init(params, rngkey=None, netstate=None):
        optstate = optimizer.init(params)
        if netstate is not None:
            netstate = jax.tree_util.tree_map(
                lambda s: jnp.repeat(s[None, ...], repeats=msharpness, axis=0), netstate
            )
        return TrainState(
            optstate=optstate, params=params, netstate=netstate, rngkey=None
        )

    def get_model_from_state(trainstate):
        return {
            "params": trainstate.params,
            "state": jax.tree_util.tree_map(lambda s: s[0], trainstate.netstate),
        }

    def _sam_gradient(lossgrad, params, netstate, X_subbatch, y_subbatch):
        new_netstate = None
        if netstate is not None:
            (loss, new_netstate), grad = lossgrad(
                params,
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            loss, grad = lossgrad(
                params,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        grad_norm = jnp.sqrt(
            sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)])
        )
        perturbed_params = jax.tree_util.tree_map(
            lambda p, g: p + rho * g / grad_norm, params, grad
        )
        if netstate is not None:
            _, perturbed_grad = lossgrad(
                perturbed_params,
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            _, perturbed_grad = lossgrad(
                perturbed_params,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        return perturbed_grad, loss, new_netstate

    def step(loss_fn, trainstate, minibatch):
        optstate = trainstate.optstate
        X_batch, y_batch = minibatch
        # split batch to simulate m-sharpness on one GPU
        X_batch = X_batch.reshape(msharpness, -1, *X_batch.shape[1:])
        y_batch = y_batch.reshape(msharpness, -1, *y_batch.shape[1:])
        grad, loss, netstate = jax.vmap(
            partial(_sam_gradient, loss_fn),
            in_axes=(None, 0, 0, 0),
            axis_name=VMAP_BATCH,
        )(trainstate.params, trainstate.netstate, X_batch, y_batch)
        grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grad)
        loss = jnp.mean(loss)

        grad = jax.lax.pmean(
            grad,
            axis_name=PMAP_BATCH,
        )

        grad, optstate = optimizer.update(grad, optstate, trainstate.params)
        params = apply_updates(trainstate.params, grad)

        newtrainstate = trainstate._replace(
            optstate=optstate,
            params=params,
            netstate=netstate,
        )

        return newtrainstate, loss

    return init, step, get_model_from_state


def build_ASAM_optimizer(
    rho: float,
    msharpness: int,
    eta: float,
    optimizer: GradientTransformation,
):
    def get_model_from_state(trainstate):
        return {
            "params": trainstate.params,
            "state": jax.tree_util.tree_map(lambda s: s[0], trainstate.netstate),
        }

    def filter_fn(name):
        return name[-1].key in ("kernel", "scale")

    def init(params, rngkey=None, netstate=None):
        optstate = optimizer.init(params)
        if netstate is not None:
            netstate = jax.tree_util.tree_map(
                lambda s: jnp.repeat(s[None, ...], repeats=msharpness, axis=0), netstate
            )
        return TrainState(
            optstate=optstate, params=params, netstate=netstate, rngkey=None
        )

    def _asam_gradient(lossgrad, params, netstate, X_subbatch, y_subbatch):
        new_netstate = None
        if netstate is not None:
            (loss, new_netstate), grad = lossgrad(
                params,
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            loss, grad = lossgrad(
                params,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        grad = jax.tree_util.tree_map_with_path(
            lambda n, g, p: g * (p + eta * jnp.sign(p)) if filter_fn(n) else g,
            grad,
            params,
        )
        grad_norm = jnp.sqrt(
            sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)])
        )
        perturbed_params = jax.tree_util.tree_map_with_path(
            lambda n, p, g: (
                p + rho * g * (p + eta * jnp.sign(p)) / grad_norm
                if filter_fn(n)
                else p + rho * g / grad_norm
            ),
            params,
            grad,
        )
        if netstate is not None:
            _, perturbed_grad = lossgrad(
                perturbed_params,
                netstate,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        else:
            _, perturbed_grad = lossgrad(
                perturbed_params,
                (X_subbatch, y_subbatch),
                is_training=True,
            )
        return perturbed_grad, loss, new_netstate

    def step(loss_fn, trainstate, minibatch):
        optstate = trainstate.optstate
        X_batch, y_batch = minibatch
        # split batch to simulate m-sharpness on one GPU
        X_batch = X_batch.reshape(msharpness, -1, *X_batch.shape[1:])
        y_batch = y_batch.reshape(msharpness, -1, *y_batch.shape[1:])
        grad, loss, netstate = jax.vmap(
            partial(_asam_gradient, loss_fn),
            in_axes=(None, 0, 0, 0),
            axis_name=VMAP_BATCH,
        )(trainstate.params, trainstate.netstate, X_batch, y_batch)
        grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grad)
        loss = jnp.mean(loss)

        grad = jax.lax.pmean(
            grad,
            axis_name=PMAP_BATCH,
        )

        grad, optstate = optimizer.update(grad, optstate, trainstate.params)
        params = apply_updates(trainstate.params, grad)

        newtrainstate = trainstate._replace(
            optstate=optstate,
            params=params,
            netstate=netstate,
        )

        return newtrainstate, loss

    return init, step, get_model_from_state
