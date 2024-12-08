import tensorflow as tf
from functools import partial
import jax.numpy as jnp
from randaugment.augment import RandAugment

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.5 * 255, 0.5 * 255, 0.5 * 255]
STDDEV_RGB = [0.5 * 255, 0.5 * 255, 0.5 * 255]
TRAINING_SIZE = 1281167
NUM_CLASSES = 1000
CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
]
CORRUPTIONS_BAR = [
    "blue_noise_sample",
    "brownish_noise",
    "caustic_refraction",
    "checkerboard_cutout",
    "cocentric_sine_waves",
    "inverse_sparkles",
    "perlin_noise",
    "plasma_noise",
    "single_frequency_greyscale",
    "sparkles",
]

import glob


def vit_inception_crop(
    image_bytes,
    image_size,
    area_range=(0.05, 1.0),
    min_object_covered=0.0,
    scope=None,
):
    original_shape = tf.io.extract_jpeg_shape(image_bytes)
    bbox = tf.zeros([0, 0, 4], tf.float32)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        original_shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        area_range=area_range,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = _resize(image, image_size)
    return image, original_shape


def _resize(image, image_size):
    return tf.image.resize(
        [image], [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC
    )[0]


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    image, original_shape = vit_inception_crop(image_bytes, image_size)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: _resize(image, image_size),
    )

    return image


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + CROP_PADDING))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [
            offset_height,
            offset_width,
            padded_center_crop_size,
            padded_center_crop_size,
        ]
    )
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = _resize(image, image_size)

    return image


def normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def mixup(images, labels, coeffs):
    new_images = images * coeffs + jnp.flip(images, 0) * (1 - coeffs)
    new_labels = labels * coeffs + jnp.flip(labels, 0) * (1 - coeffs)
    return new_images, new_labels


DEFAULT_RAND_AUGMENT = dict(num_layers=2, magnitude=10.0)


def preprocess_for_train(
    image_bytes,
    dtype=tf.float32,
    image_size=IMAGE_SIZE,
    rand_augment=False,
    rand_augment_options=DEFAULT_RAND_AUGMENT,
):
    """Preprocesses the given image for training.

    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

    Returns:
    A preprocessed image `Tensor`.
    """
    image = _decode_and_random_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    if rand_augment:
        image = RandAugment(**rand_augment_options).distort(image)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(
    image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, before_norm=False
):
    """Preprocesses the given image for evaluation.

    Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

    Returns:
    A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    if before_norm:
        return tf.image.convert_image_dtype(image / 255.0, dtype=dtype)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def decode_training_example(example, dtype, rand_augment, rand_augment_options):
    feature_description = {
        "image/height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/channels": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/class/label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/class/synset": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/class/text": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/format": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/filename": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    return {
        "image": preprocess_for_train(
            parsed_example["image/encoded"],
            dtype=dtype,
            rand_augment=rand_augment,
            rand_augment_options=rand_augment_options,
        ),
        "label": parsed_example["image/class/label"] - 1,
    }


def decode_eval_example(example, dtype, before_norm=False):
    feature_description = {
        "image/height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/channels": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/class/label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/class/synset": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/class/text": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/format": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/filename": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    return {
        "image": preprocess_for_eval(
            parsed_example["image/encoded"], dtype=dtype, before_norm=before_norm
        ),
        "label": parsed_example["image/class/label"] - 1,
    }


def get_training_loader(
    root,
    batch_size,
    dtype=tf.float32,
    shuffle_buffer_size=2048,
    cache=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch=tf.data.AUTOTUNE,
    repeat=-1,
    drop_remainder=True,
    rand_augment=False,
    rand_augment_options=DEFAULT_RAND_AUGMENT,
):
    filenames = sorted(glob.glob(f"{root}/train-*-of-01024"))
    filenames = tf.convert_to_tensor(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames).interleave(
        lambda name: tf.data.TFRecordDataset(name),
        cycle_length=None,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
        block_length=None,
    )
    if cache:
        ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if repeat > 0:
        ds = ds.repeat(repeat)
    ds = ds.map(
        partial(
            decode_training_example,
            dtype=dtype,
            rand_augment_options=rand_augment_options,
            rand_augment=rand_augment,
        ),
        num_parallel_calls=num_parallel_calls,
    )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(prefetch)
    return ds


def get_eval_loader(
    root,
    batch_size,
    dtype,
    cache=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch=tf.data.AUTOTUNE,
    before_norm=False,
):
    filenames = sorted(glob.glob(f"{root}/validation-*-of-00128"))
    filenames = tf.convert_to_tensor(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames).interleave(
        lambda name: tf.data.TFRecordDataset(name),
        cycle_length=None,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
        block_length=None,
    )
    if cache:
        ds = ds.cache()
    ds = ds.map(
        partial(decode_eval_example, dtype=dtype, before_norm=before_norm),
        num_parallel_calls=num_parallel_calls,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch)
    return ds


def decode_corrupt_example(example):
    feature_description = {
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "label_name": tf.io.FixedLenFeature([], dtype=tf.string),
        "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = normalize_image(
        tf.cast(tf.io.decode_jpeg(parsed_example["image_raw"]), tf.float32)
    )
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return {"image": image, "label": parsed_example["label"]}


def decode_corrupt_example_256(example, num_label_per_example=1):
    feature_description = {
        "image/height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/channels": tf.io.FixedLenFeature([], dtype=tf.int64),
        "image/class/label": tf.io.FixedLenFeature(
            [num_label_per_example] if num_label_per_example > 1 else [], dtype=tf.int64
        ),
        "image/class/synset": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/class/text": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/format": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/filename": tf.io.FixedLenFeature([], dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(parsed_example["image/encoded"])
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.central_crop(image, 224 / 256)
    image = normalize_image(tf.cast(image, tf.float32))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return {"image": image, "label": parsed_example["image/class/label"] - 1}


def get_corrupted_loader(
    path,
    batch_size,
    cache=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch=tf.data.AUTOTUNE,
):
    ds = tf.data.TFRecordDataset(path)
    if cache:
        ds = ds.cache()
    ds = ds.map(decode_corrupt_example, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch)
    return ds


def get_corrupted_loader_256(
    path,
    batch_size,
    cache=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch=tf.data.AUTOTUNE,
    num_label_per_example=1,
):
    ds = tf.data.TFRecordDataset(path)
    if cache:
        ds = ds.cache()
    ds = ds.map(
        partial(
            decode_corrupt_example_256, num_label_per_example=num_label_per_example
        ),
        num_parallel_calls=num_parallel_calls,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch)
    return ds


def get_corrupted_bar_loader(
    path,
    batch_size,
    dtype,
    cache=False,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch=tf.data.AUTOTUNE,
):
    ds = tf.data.TFRecordDataset(path)
    if cache:
        ds = ds.cache()
    ds = ds.map(
        partial(decode_eval_example, dtype=dtype), num_parallel_calls=num_parallel_calls
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch)
    return ds
