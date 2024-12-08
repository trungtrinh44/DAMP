import os

corruptions = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
]
print("Make sure that you are inside the data/ImageNet folder")
os.makedirs(f"../ImageNet-C", exist_ok=True)
for corruption in corruptions:
    for i in range(1, 6):
        os.makedirs(f"ImageNet-C/{corruption}_{i}", exist_ok=True)
        os.system(
            f"python build_imagenet_data.py -validation_directory ImageNet-C/{corruption}/{i} -output_directory ImageNet-C/{corruption}_{i}"
        )
        os.system(
            f"mv ImageNet-C/{corruption}_{i}/validation-00000-of-00001 ../ImageNet-C/{corruption}_{i}.tfrecords"
        )
os.system("rm -rf ImageNet-C")
print("Make sure that there is a data/ImageNet-C folder with 75 tfrecords file in it.")
