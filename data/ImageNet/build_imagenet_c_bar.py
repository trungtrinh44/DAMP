import os

corruptions = [
    "sparkles",
    "plasma_noise",
    "blue_noise_sample",
    "brownish_noise",
    "caustic_refraction",
    "checkerboard_cutout",
    "cocentric_sine_waves",
    "inverse_sparkles",
    "perlin_noise",
    "single_frequency_greyscale",
]
print("Make sure that you are inside the data/ImageNet folder")
os.makedirs(f"../ImageNet-C-bar", exist_ok=True)
for corruption in corruptions:
    for i in range(1, 6):
        os.makedirs(f"ImageNet-C-bar/{corruption}_{i}", exist_ok=True)
        os.system(
            f"python build_imagenet_data.py -validation_directory ImageNet-C-bar/{corruption}/{i} -output_directory ImageNet-C-bar/{corruption}_{i}"
        )
        os.system(
            f"mv ImageNet-C-bar/{corruption}_{i}/validation-00000-of-00001 ../ImageNet-C-bar/{corruption}_{i}.tfrecords"
        )
os.system("rm -rf ImageNet-C-bar")
print(
    "Make sure that there is a data/ImageNet-C-bar folder with 50 tfrecords file in it."
)
