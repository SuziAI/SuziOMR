import copy
import cv2
import dataclasses
from suzipu import split_training_validation_by_class_suzipu
from lvlvpu import split_training_validation_by_class_lvlvpu
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.util import random_noise
import torch
import torchvision.transforms as transforms

BATCH_SIZE = 100


def shrink(is_random=True, target_size=20):
    def inner(input_image):

        t_size = random.randint(int(0.75 * target_size), int(target_size + 2)) if is_random else target_size

        original_width = input_image.shape[-1]
        original_height = input_image.shape[-2]
        aspect_ratio = original_width / original_height * random.uniform(0.6,
                                                                         1.5) if is_random else original_width / original_height

        if aspect_ratio > 1:
            w = int(t_size)
            h = int(t_size / aspect_ratio)
        else:
            w = int(t_size * aspect_ratio)
            h = int(t_size)

        output_image = transforms.Resize(size=(h, w), interpolation=transforms.InterpolationMode.NEAREST_EXACT)(
            input_image)
        return output_image

    return inner


def paste_to_square(is_random=True, target_size=28):
    def inner(input_image):
        ## Modify the function to extend the
        ## input image to a square of 40x40.
        ## Tip: This can be done by clever use
        ## of the Pad function
        ## https://pytorch.org/vision/stable/generated/torchvision.transforms.Pad.html
        ## Also make sure that the added padding
        ## on each side is random, i.e., the
        ## data itself is augmented by its
        ## position in the square.

        pad_width = target_size - input_image.shape[-1]
        pad_height = target_size - input_image.shape[-2]

        left_pad = random.randint(0, pad_width) if is_random else pad_width // 2
        top_pad = random.randint(0, pad_height) if is_random else pad_height // 2

        right_pad = pad_width - left_pad
        bottom_pad = pad_height - top_pad

        output_image = transforms.Pad(padding=(left_pad, top_pad, right_pad, bottom_pad), fill=1)(input_image)
        return output_image

    return inner


def salt_and_pepper(percentage=0.1, amount=0.001):
    def inner(input_image):
        output_image = input_image.numpy().squeeze()
        if random.uniform(0, 1) < percentage / 2:
            output_image = random_noise(output_image, mode='salt', amount=2 * amount)
        if random.uniform(0, 1) < percentage / 2:
            output_image = random_noise(output_image, mode='pepper', amount=amount)
        return torch.Tensor(output_image).unsqueeze(0)

    return inner


def erode(percentage=0.1):
    def inner(input_image):
        if random.uniform(0, 1) < percentage:  # only apply transformation according to percentage
            kernel = np.ones((2, 2), np.uint8)
            output_image = cv2.erode(input_image, kernel, iterations=1)
            return output_image
        else:
            return input_image

    return inner


def dilate(percentage=0.1):
    def inner(input_image):
        if random.uniform(0, 1) < percentage:  # only apply transformation according to percentage
            kernel = np.ones((2, 2), np.uint8)
            output_image = cv2.dilate(input_image, kernel, iterations=1)
            return output_image
        else:
            return input_image

    return inner


def get_cropped_dataset(dataset, remove_blobs=True):  ## creates a new dataset where each of the images is cropped
    def remove_small_blobs(image):  # remove small isolated connected black areas
        noise_removal_threshold = 1
        mask = np.ones_like(image)*255
        contours, hierarchy = cv2.findContours(255-image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
          area = cv2.contourArea(contour)
          if area >= noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 0)
        return mask


    def crop_excess_whitespace(image):
        gray = 255*(image < 128).astype(np.uint8) #reverse the colors
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        rect = image[y:y+h, x:x+w]
        return rect

    cropped_dataset = copy.deepcopy(dataset)
    for idx in range(len(cropped_dataset)):
        cropped_dataset[idx]["image"] = crop_excess_whitespace(remove_small_blobs(dataset[idx]["image"])) if remove_blobs else crop_excess_whitespace(dataset[idx]["image"])
    return cropped_dataset


@dataclass
class Editions:  # This parameter chooses which edition is used for validation. The other editions are for training.
    NONE: str = "none"
    LU: str = "lu"
    ZHANG: str = "zhang"
    SIKU: str = "siku"
    ZHU: str = "zhu"
    SHANGHAI: str = "shanghai"


@dataclass
class LabelType:
    PITCH_BALANCED: str = "pitch_balanced"
    SECONDARY_BALANCED: str = "secondary_balanced"
    NONE: str = "none"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_type, dataset: list, transform=None):
        self.X = [entry["image"] for entry in dataset]

        if "pitch" in label_type:
            self.y = [entry["annotation"]["pitch"] for entry in dataset]
        elif "secondary" in label_type:
            self.y = [entry["annotation"]["secondary"] for entry in dataset]
        else:
            self.y = [entry["annotation"] for entry in dataset]

        try:
            self.is_simple = [entry["is_simple"] for entry in dataset]
        except KeyError:
            pass

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        label = self.y[idx]

        try:
            is_simple = self.is_simple[idx]
            return sample, label, is_simple
        except AttributeError:
            return sample, label


def get_dataloaders(dataset, transformations, validation_edition=Editions.NONE, test_edition=Editions.SIKU, label_type=None,
                    artificial_dataset=[], image_size=28, is_suzipu=True):
    get_train_transforms = transformations["train"]
    get_validation_transforms = transformations["validation"]
    get_test_transforms = transformations["test"]

    if is_suzipu:
        split_training_validation_by_class = split_training_validation_by_class_suzipu
    else:
        split_training_validation_by_class = split_training_validation_by_class_lvlvpu
        label_type = ""

    def get_datasets():
        if validation_edition is not Editions.NONE:
            train_entries = [entry for entry in dataset if entry["edition"] not in (validation_edition, test_edition)]
            if artificial_dataset:
                train_entries += artificial_dataset
            train_data = Dataset(label_type, train_entries, transform=get_train_transforms(image_size=image_size))
            validation_data = Dataset(label_type,
                                      [entry for entry in dataset if entry["edition"] == validation_edition],
                                      transform=get_validation_transforms(image_size=image_size))
            test_data = Dataset(label_type, [entry for entry in dataset if entry["edition"] == test_edition],
                                transform=get_test_transforms(image_size=image_size))
        else:
            td_data = [entry for entry in dataset if entry["edition"] != test_edition]
            train_entries, validation_entries = split_training_validation_by_class(td_data, 0.75)
            if artificial_dataset:
                train_entries += artificial_dataset
            train_data = Dataset(label_type, train_entries, transform=get_train_transforms(image_size=image_size))
            validation_data = Dataset(label_type, validation_entries,
                                      transform=get_validation_transforms(image_size=image_size))
            test_data = Dataset(label_type, [entry for entry in dataset if entry["edition"] == test_edition],
                                transform=get_test_transforms(image_size=image_size))
        return train_data, validation_data, test_data

    def get_dataloaders(train_data, validation_data, test_data):
        def get_sampler(y):
            inverse_class_weights = 1 / np.unique(y, return_counts=True)[1]
            inverse_weigths = [inverse_class_weights[int(label)] for label in y]

            return torch.utils.data.WeightedRandomSampler(weights=inverse_weigths, num_samples=len(y), replacement=True)

        sample = "balanced" in label_type

        train_sampler = get_sampler(train_data.y) if sample else None
        validation_sampler = None

        loaders = {
            'train': torch.utils.data.DataLoader(train_data,
                                                 batch_size=BATCH_SIZE,
                                                 sampler=train_sampler),

            'validation': torch.utils.data.DataLoader(validation_data,
                                                      batch_size=BATCH_SIZE,
                                                      sampler=validation_sampler),
            'test': torch.utils.data.DataLoader(test_data,
                                                batch_size=BATCH_SIZE),
        }

        return loaders

    train_data, validation_data, test_data = get_datasets()

    return get_dataloaders(train_data, validation_data, test_data)


def get_all_dataloaders(dataset, transformations, test_edition=Editions.SHANGHAI, artificial_dataset=[], image_size=48, is_suzipu=True):
    val_edition = Editions.NONE

    dataloader_dict = {}

    for label_type in dataclasses.astuple(LabelType()):
        dataloader_dict[label_type] = get_dataloaders(dataset, validation_edition=val_edition, test_edition=test_edition,
                                                   label_type=label_type,
                                                   artificial_dataset=artificial_dataset,
                                                   image_size=image_size,
                                                   transformations=transformations,
                                                   is_suzipu=is_suzipu)
    dataloader_dict["full"] = get_dataloaders(dataset, validation_edition=val_edition, test_edition=test_edition,
                                                   label_type="", artificial_dataset=artificial_dataset,
                                                   image_size=image_size,
                                                   transformations=transformations,
                                                   is_suzipu=is_suzipu)
    return dataloader_dict


def visualize_dataloader(dl, label_to_string=None):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dl.dataset), size=(1,)).item()

        try:
            img, label, _ = dl.dataset[sample_idx]
        except ValueError:
            img, label = dl.dataset[sample_idx]

        figure.add_subplot(rows, cols, i)

        if label_to_string is not None:
            label = label_to_string[label]

        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()