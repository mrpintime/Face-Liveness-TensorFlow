import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import tensorflow as tf

class Data_Loader:
    def __init__(
        self, target_size=(224, 224), augmentation=False, rescale=1.0 / 255
    ) -> None:
        self.target_size = target_size
        self.augmentation = augmentation
        self.rescale = rescale

    def load_images_from_directory(self, directory):
        image_data = []
        labels = []

        # Augmentation configuration
        if self.augmentation:
            seq = iaa.Sequential(
                [
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    iaa.Sometimes(
                        0.5, iaa.GaussianBlur(sigma=(0, 0.5))
                    ),  # Gaussian blur with random sigma
                    iaa.ContrastNormalization((0.75, 1.5)),  # contrast normalization
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),  # additive Gaussian noise
                    iaa.Multiply(
                        (0.8, 1.2), per_channel=0.2
                    ),  # multiply each pixel with random values
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scaling of images
                        translate_percent={
                            "x": (-0.2, 0.2),
                            "y": (-0.2, 0.2),
                        },  # translation of images
                        rotate=(-25, 25),  # rotation of images
                        shear=(-8, 8),  # shearing of images
                    ),
                ],
                random_order=True,
            )  # apply augmentations in random order

        # Iterate over each class directory
        for label, class_name in enumerate(os.listdir(directory)):
            class_dir = os.path.join(directory, class_name)

            # Iterate over each image in the class directory
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)
                image = image.resize(self.target_size)
                image = np.array(image)

                if self.augmentation:
                    # Apply augmentation
                    image_augmented = seq.augment_image(image)
                    image_data.append(image_augmented)
                else:
                    image_data.append(image)

                # Apply rescaling
                image_data[-1] = image_data[-1] * self.rescale
                labels.append(label)

        return np.array(image_data), np.array(labels)

    # prepare loaded images for model
    def prepare_data(self, img, all_label, smoothing=True):
        images = []
        labels = []
        masks = []
        label_weight = 0.99 if smoothing else 1.0
        for image, label in zip(img, all_label):
            label_tensortype = tf.cast(label, dtype=tf.float32)
            label = label_tensortype
            map_size = 14  # this is size of our feature mao that will produce in model
            if label == 0:
                mask = np.ones((1, map_size, map_size), dtype=np.float32) * (
                    1 - label_weight
                )
            else:
                mask = np.ones((1, map_size, map_size), dtype=np.float32) * (
                    label_weight
                )

            images.append(image)
            labels.append(label)
            masks.append(mask)
        # change list to numpy array
        x_train = np.array(images)
        y_train_mask = np.array(masks)
        y_train_label = np.array(labels)
        # prepare dimension
        y_train_mask = np.squeeze(y_train_mask, axis=1)
        y_train_mask = np.expand_dims(y_train_mask, axis=-1)
        y_train_label = np.expand_dims(y_train_label, axis=-1)

        return x_train, y_train_label, y_train_mask
