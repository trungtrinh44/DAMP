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

# Preparing the ImageNet-A dataset

Inside the folder `data/ImageNet`, run the following command to download and extract the tarfile:
```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvf imagenet-a.tar
rm -rf imagenet-a.tar
```
Now we have a folder `imagenet-a` inside the `data/ImageNet` folder. Then activate the installed python environment (to install the environment, read `install_steps.md` in the main directory) and run the following command:
```bash
python build_imagenet_a.py -validation_directory imagenet-a -output_directory imagenet-a
```
which will create the `validation-00000-of-00001` file inside the `imagenet-a` folder. Next we copy this file to the `data` folder:
```bash
mv imagenet-a/validation-00000-of-00001 ../ImageNet-A.tfrecord
```
then we can remove the `imagenet-a` folder:
```bash
rm -rf imagenet-a
```

Make sure that we now have the `ImageNet-A.tfrecord` file inside the `data` folder.

# Preparing the ImageNet-D dataset

Downlaod the tarfile from the [link](https://drive.google.com/file/d/11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc/view) and move it into the `data/ImageNet` folder. Inside `data/ImageNet`, run the following command to extract and delete the tarfile:
```bash
tar -xvf ImageNet-D.tar
```
This will create a new `ImageNet-D` folder inside the `data/ImageNet` folder. Next run the following commands to create the `tfrecord`:
```bash
python build_imagenet_d.py -validation_directory ImageNet-D/background -output_directory ImageNet-D/background

python build_imagenet_d.py -validation_directory ImageNet-D/material -output_directory ImageNet-D/material

python build_imagenet_d.py -validation_directory ImageNet-D/texture -output_directory ImageNet-D/texture
```

Then run the following commands to copy the `tfrecord` files to the desired folder:
```bash
mkdir ../ImageNet-D/
mv ImageNet-D/background/validation-00000-of-00001 ../ImageNet-D/background.tfrecord
mv ImageNet-D/material/validation-00000-of-00001 ../ImageNet-D/material.tfrecord
mv ImageNet-D/texture/validation-00000-of-00001 ../ImageNet-D/texture.tfrecord
```

Finally we can remove the `ImageNet-D` folder
```bash
rm -rf ImageNet-D
```

Make sure that we now have the `data/ImageNet-D` folder with 3 tfrecord files `background.tfrecord`, `material.tfrecord`, and `texture.tfrecord`.