import jax
from optax import (
    chain,
    scale_by_adam,
    add_decayed_weights,
    clip_by_global_norm,
    scale_by_learning_rate,
    trace,
)


def sample_like_tree(a, key):
    """get a random gaussian variable for every parameter in tree"""
    treedef = jax.tree_util.tree_structure(a)
    num_vars = len(jax.tree_util.tree_leaves(a))
    key, *subkeys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.normal(k, shape=p.shape),
        a,
        jax.tree_util.tree_unflatten(treedef, subkeys),
    )
    return noise, key


def adamw(
    learning_rate,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    eps_root=0.0,
    weight_decay=1e-4,
    max_norm=1.0,
    mask=None,
):
    return chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=None, nesterov=False
        ),
        add_decayed_weights(weight_decay, mask),
        scale_by_learning_rate(learning_rate),
    )


def nesterov(learning_rate, momentum, weight_decay, mask=None):
    return chain(
        add_decayed_weights(weight_decay, mask),
        trace(decay=momentum, nesterov=True),
        scale_by_learning_rate(learning_rate),
    )
