import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from src.utils import seed_worker


def resize_and_crop(image, target_size= (224,224) ):
    """
    Resize and centre-crop a multi-channel image while
    preserving its aspect ratio.

    Parameters
    ---------
    image : tensor
        Input tensor of shape (3, 1024, 1280)
    target_size : vector, default=(240, 240)
        Target size of the cropped image.

    Returns
    -------
    Image_cropped : tensor
        Image of shape (3, 240, 240)
    """
    c, h , w = image.shape
    scale = max( target_size[0] / h , target_size[1] / w )
    new_h, new_w = int(h * scale), int(w * scale)

    image_batched = image.unsqueeze(0)

    image_resized = F.interpolate(
        image_batched,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

    image_resized = image_resized.squeeze(0)

    start_h = (new_h - target_size[0]) // 2
    start_w = (new_w - target_size[1]) // 2

    image_cropped = image_resized[
        :,
        start_h : start_h + target_size[0],
        start_w : start_w + target_size[1]
    ]

    return image_cropped

def normalise_fluorescence_channels(image, percentile_low=1, percentile_high=99):
    """
    Normalise the fluorescence channels of a multi-channel image independently.

    Parameters
    ----------
    image : tensor
        Input tensor of shape (3, H, W) where the channels are [DAPI, Tubulin, Actin].
    percentile_low : int, default=1
        Lower percentile value for clipping.
    percentile_high : int, default=99
        Upper percentile value for clipping.

    Returns
    -------
    Normalised_image
        tensor of shape (3, H, W)

    """
    normalised_image = image.clone()

    for c in range(image.shape[0]):
        channel = normalised_image[c]
        flat_channel = channel.flatten()
        p_low = np.percentile(flat_channel.numpy(), percentile_low)
        p_high = np.percentile(flat_channel.numpy(), percentile_high)

        channel = torch.clamp(channel, min=p_low, max=p_high)

        if p_high > p_low:
            normalised_image[c] = (channel - p_low) / (p_high - p_low)
        else:
            normalised_image[c] = torch.zeros_like(channel)

    return normalised_image

class ImageAugmentation:
    """
    Data augmentation with flips and 90-degree rotations.
    """
    def __init__(self,
                 flip_h_prob=0.5,
                 flip_v_prob=0.5,
                 rotate_prob=0.5
                 ):
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.rotate_prob = rotate_prob

    def __call__(self, image):
        """
        Apply augmentation to image.

        Parameters
        ---------
        image
            Tensor of shape (3, H, W)

        Returns
        --------
        augmented_image
            Augmented tensor of shape (3, H, W)
        """
        if random.random() < self.flip_h_prob:
            image = torch.flip(image, dims=[-1])

        if random.random() < self.flip_v_prob:
            image = torch.flip(image, dims=[-2])

        if random.random() < self.rotate_prob:
            k = random.randint(0, 3)
            image = torch.rot90(image, k=k, dims=[-2, -1])

        return image

train_transform = Compose([
    resize_and_crop,
    normalise_fluorescence_channels,
    ImageAugmentation(),
])

eval_transform = Compose([
    resize_and_crop,
    normalise_fluorescence_channels
])

class TransformedSubset(Subset):
    """
    Subset of a dataset that applies a specified transformation to each image.

    This allows for different transformations to be applied to the training,
    validation, and test subsets while sharing the same underlying dataset.

    Parameters
    ---------
    dataset : BBC021Dataset
        Entire BBC021Dataset.
    indices : list[int]
        Indices of images included in the subset.
    transform : callable or None, default=None
        Transformation applied to each image when it is retrieved.
    """
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

def create_dataloaders(dataset, train_indices, val_indices, test_indices,
                       batch_size=32, num_workers=4, seed=42):
    """
    Create training and validation dataloaders.

    Parameters
    ---------
    dataset : BBC021Dataset
        Entire BBC021Dataset.
    train_indices : list[int]
        Indices of images from the training set.
    val_indices : list[int]
        Indices of images from the validation set.
    test_indices : list[int]
        Indices of images from the test set.
    batch_size : int, default=32
        Number of samples per batch.
    num_workers : int, default=4
        Number of subprocesses for data loading.

    Returns
    -------
    train_loader : DataLoader
        Data loader for the training set.
    val_loader : DataLoader
        Data loader for the validation set.
    test_loader : DataLoader
        Data loader for the test set.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    train_subset = TransformedSubset(
        dataset,
        train_indices,
        transform=train_transform
    )

    val_subset = TransformedSubset(
        dataset,
        val_indices,
        transform=eval_transform
    )

    test_subset = TransformedSubset(
        dataset,
        test_indices,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader



