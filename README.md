# Data augmentation via multiplicative perturbations

This repository contains a Jax/Flax implementation of the paper

[Improving robustness to corruptions with multiplicative weight perturbations](https://openreview.net/forum?id=M8dy0ZuSb1)

by Trung Trinh, Markus Heinonen, Luigi Acerbi and Samuel Kaski

For more information about the paper, please visit the [website](https://trungtrinh44.github.io/DAMP/).

Please cite our work if you find it useful:

```bibtex
@inproceedings{trinh2024improving,
    title={Improving robustness to corruptions with multiplicative weight perturbations},
    author={Trung Trinh and Markus Heinonen and Luigi Acerbi and Samuel Kaski},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=M8dy0ZuSb1}
}
```
## Setting up

### Installing the python packages
Please follow the instructions in `install_steps.md` to install the necessary python packages.

### Downloading the small datasets
To run the experiments for CIFAR-10/100 and TinyImageNet, one needs to run the following commands to download the necessary datasets and store them in the `data` folder:
```bash
bash download_scripts/download_cifar10_c.sh
bash download_scripts/download_cifar100_c.sh
bash download_scripts/download_tinyimagenet.sh
bash download_scripts/download_tinyimagenet_c.sh
```

### ImageNet experiments
To set up the ImageNet experiments, please follow the instructions in `README.md` in the `data/ImageNet` folder.

## Instructions to replicate the results

To replicate an experiment, please use the corresponding JSON config file in the `configs` folder. Please use the `train.py` script for CIFAR-10/100 and TinyImageNet experiments. The syntax is:

```bash
python train.py --conf ${PATH_TO_CONFIG} --experiment_id ${EXP_ID} --seed ${RANDOM_SEED} --base_dir ${BASE_DIR}
```
The experiment will be stored in the folder `${BASE_DIR}/${TRAINING_METHOD}/${MODEL}/${DATASET}/${EXP_ID}`

For example, to replicate the result of DAMP on CIFAR-10, run the following command:
```bash
python train.py --conf configs/damp/resnet18_cifar10.json --experiment_id 1 --seed 44 --base_dir experiments
```
and the experiment can be found at `experiments/DAMP/ResNet18/cifar10/1`.