# Preparing the ImageNet dataset

First, download the ImageNet dataset, which includes two tar files `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`. Copy these two files into this folder (`data/ImageNet`).

Then inside `data/ImageNet`, run the `extract_imagenet.sh` script to extract the training and validation data into the temporary `train` and `val` folders inside the `ImageNet` folder.

Then activate the installed python environment (to install the environment, read `install_steps.md` in the main directory) and run the following commands:
```bash
python build_imagenet_data.py -train_directory train -output_directory ${pwd}
python build_imagenet_data.py -validation_directory val -output_directory ${pwd}
```

Finally, run the following commands to remove unnecessary folders and files:
```bash
rm -rf train & rm -rf val & rm -rf ILSVRC2012_img_train.tar & rm -rf ILSVRC2012_img_val.tar
```

Inside the `ImageNet` folder is now the following files:
`train-00000-of-01024` to `train-01023-of-01024` which are `tfrecord` files containing the training set, and `validation-00000-of-00128` to `validation-00127-of-00128` files containing the validation set.

TODO: add instructions to prepare the corrupted test sets of ImageNet.