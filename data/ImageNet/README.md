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
rm -rf train && rm -rf val && rm -rf ILSVRC2012_img_train.tar && rm -rf ILSVRC2012_img_val.tar
```

Inside the `ImageNet` folder is now the following files:
`train-00000-of-01024` to `train-01023-of-01024` which are `tfrecord` files containing the training set, and `validation-00000-of-00128` to `validation-00127-of-00128` files containing the validation set.

TODO: add instructions to prepare the corrupted test sets of ImageNet.

# Preparing the corrupted test sets of ImageNet

## Preparing the ImageNet-C dataset

Download the dataset from [link](https://github.com/hendrycks/robustness) and move the tar files to `data/ImageNet`, then extract and delete the files:
```bash
mkdir ImageNet-C
tar -xvf blur.tar -C ImageNet-C && rm -rf blur.tar
tar -xvf digital.tar -C ImageNet-C && rm -rf digital.tar
tar -xvf extra.tar -C ImageNet-C && rm -rf extra.tar
tar -xvf noise.tar -C ImageNet-C && rm -rf noise.tar
tar -xvf weather.tar -C ImageNet-C && rm -rf weather.tar
```

Then activate the installed python environment (to install the environment, read `install_steps.md` in the main directory) and run the following command:
```bash
python build_imagenet_c.py
```

Make sure that there is now a `data/ImageNet-C` folder with 75 tfrecords files in it.


## Preparing the ImageNet-C-bar dataset

Run the following command to download the dataset
```bash
wget https://dl.fbaipublicfiles.com/inc_bar/imagenet_c_bar.tar.gz
```
then move the file to `data/ImageNet`, then extract and delete the file:
```bash
tar -xvf imagenet_c_bar.tar.gz && rm -rf imagenet_c_bar.tar.gz
mv cbar_download ImageNet-C-bar
```

Then activate the installed python environment (to install the environment, read `install_steps.md` in the main directory) and run the following command:
```bash
python build_imagenet_c_bar.py
```

Make sure that there is now a `data/ImageNet-C-bar` folder with 50 tfrecords files in it.


## Preparing the ImageNet-A dataset

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

## Preparing the ImageNet-D dataset

Download the tarfile from the [link](https://drive.google.com/file/d/11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc/view) and move it into the `data/ImageNet` folder. Inside `data/ImageNet`, run the following command to extract and delete the tarfile:
```bash
tar -xvf ImageNet-D.tar && rm -rf ImageNet-D.tar
```
This will create a new `ImageNet-D` folder inside the `data/ImageNet` folder. Next run the following commands to create the `tfrecord`:
```bash
python build_imagenet_d.py -validation_directory ImageNet-D/background -output_directory ImageNet-D/background

python build_imagenet_d.py -validation_directory ImageNet-D/material -output_directory ImageNet-D/material

python build_imagenet_d.py -validation_directory ImageNet-D/texture -output_directory ImageNet-D/texture
```

Then run the following commands to move the `tfrecord` files to the desired folder:
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

## Preparing the ImageNet-Sketch dataset

Download the zip file from the [link](https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view) and move it into the `data/ImageNet` folder. Inside this folder, run the following command to extract and delete the zip file:
```bash
unzip ImageNet-Sketch.zip && rm -rf ImageNet-Sketch.zip
```

We now have a `sketch` folder inside the `data/ImageNet` folder. Now run the following command to build the `tfrecord` file:
```bash
python build_imagenet_data.py -validation_directory sketch -output_directory sketch
```
Then run the following commands to move the `tfrecord` files to the desired folder:
```bash
mv sketch/validation-00000-of-00001 ../ImageNet-Sketch.tfrecord
```

Make sure that we now have a `ImageNet-Sketch.tfrecord` file inside the `data` folder. Finally we can remove the `sketch` folder:
```bash
rm -rf sketch
```

## Preparing the ImageNet-Drawing and ImageNet-Cartoon

Inside the `data/ImageNet` folder, run the following commands to download both datasets:
```bash
wget https://zenodo.org/records/6801109/files/imagenet-cartoon.tar.gz?download=1
wget https://zenodo.org/records/6801109/files/imagenet-drawing.tar.gz?download=1
```

Then extract and delete the tar files:
```bash
tar -xvf imagenet-cartoon.tar.gz && rm -rf imagenet-cartoon.tar.gz
tar -xvf imagenet-drawing.tar.gz && rm -rf imagenet-drawing.tar.gz
```

We now have a `datasets` folder containing two folders `imagenet-cartoon` and `imagenet-drawing`. Next we build the tfrecord files:
```bash
python build_imagenet_data.py -validation_directory datasets/imagenet-cartoon -output_directory datasets/imagenet-cartoon

python build_imagenet_data.py -validation_directory datasets/imagenet-drawing -output_directory datasets/imagenet-drawing
```

Then we move the files to the desired location:
```bash
mv datasets/imagenet-cartoon/validation-00000-of-00001 ../ImageNet-Cartoon.tfrecord
mv datasets/imagenet-drawing/validation-00000-of-00001 ../ImageNet-Drawing.tfrecord
```

Make sure that we now have the files `ImageNet-Cartoon.tfrecord` and `ImageNet-Drawing.tfrecord` inside the `data` folder. Then we can remove the redundance folders:
```bash
rm -rf datasets
```