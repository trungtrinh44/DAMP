import jax
import jax.numpy as jnp
import numpy as np


class LrScheduler:
    def __init__(
        self,
        init_value,
        num_epochs,
        milestones,
        lr_ratio,
        num_start_epochs,
        step_per_epoch,
    ):
        self.init_value = init_value
        self.num_epochs = num_epochs
        self.milestones = milestones
        self.lr_ratio = lr_ratio
        self.num_start_epochs = num_start_epochs
        self.step_per_epoch = step_per_epoch
        self._lrs = jnp.array([self.__lr(i) for i in range(self.num_epochs)])

    def __call__(self, step_count):
        return self._lrs[step_count // self.step_per_epoch]

    def __lr(self, epoch):
        if epoch < self.num_start_epochs:
            return (
                self.init_value * (1.0 - self.lr_ratio) / self.num_start_epochs * epoch
                + self.lr_ratio
            )
        t = epoch / self.num_epochs
        m1, m2 = self.milestones
        if t <= m1:
            factor = 1.0
        elif t <= m2:
            factor = 1.0 - (1.0 - self.lr_ratio) * (t - m1) / (m2 - m1)
        else:
            factor = self.lr_ratio
        return self.init_value * factor


def ece(softmax_logits, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(softmax_logits, -1), np.argmax(softmax_logits, -1)
    accuracies = predictions == labels

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)

    return ece
