import os
import torch
from torch import nn, Tensor
from typing import Any
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for our testing purposes, please do not change!


def _add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def add_sn_(model: nn.Module):
    model.apply(_add_sn)


def get_sd_map_from_tensor(
        tensor: Tensor,
        num_spatial_classes: int = 2,
        tensor_shape: tuple = (224, 224),
        class_index: Tensor = 0,
) -> Tensor:
    """从掩码张量中获取空间分类张量

    Args:
        tensor (Tensor): 掩码张量
        num_spatial_classes (int, optional): 空间层数, 如果只有前景和背景, 则为2, 默认: 2
        tensor_shape (tuple, optional): 张量尺寸, 默认: (224, 224)
        class_index (Tensor, optional): 类别索引, 默认: 0
    """

    sd_map_tensor = torch.zeros(num_spatial_classes, tensor_shape[0], tensor_shape[1])
    sd_map_tensor[class_index] = tensor[0]

    return sd_map_tensor


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()


def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    #### START CODE HERE ####
    return F.one_hot(labels, n_classes)
    #### END CODE HERE ####


def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    # Note: Make sure this function outputs a float no matter what inputs it receives
    #### START CODE HERE ####
    combined = torch.cat((x.float(), y.float()), 1)
    #### END CODE HERE ####
    return combined


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns:
        generator_input_dim: the input dimensionality of the conditional generator,
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    #### START CODE HERE ####
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    #### END CODE HERE ####
    return generator_input_dim, discriminator_im_chan
