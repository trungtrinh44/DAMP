Using Mamba https://mamba.readthedocs.io/en/latest/ to manage the packages. After installing Mamba, run the following commands on a machine with Nvidia GPUs to install the python packages.
```bash
mamba create -n jax_flax python==3.11
mamba activate jax_flax
mamba install jaxlib=*=*cuda* jax -c conda-forge
mamba install cuda-nvcc -c nvidia
mamba install scikit-learn
mamba install pytorch torchvision cpuonly -c pytorch
pip install chex==0.1.86 optax==0.2.2 dm-tree==0.1.8 flax==0.8.2 dm-haiku==0.0.11 tensorflow==2.16.1 tensorflow-datasets==4.9.4
```bash