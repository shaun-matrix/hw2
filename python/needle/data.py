import numpy as np
import gzip
import struct
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img, axis=1)
            return img
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        h, w, _ = img.shape
        pad_img = np.pad(img, ((self.padding,self.padding), \
          (self.padding,self.padding), (0,0)), 'constant')
        return pad_img[self.padding+shift_x:h+self.padding+shift_x, \
          self.padding+shift_y:w+self.padding+shift_y, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            dataset_indexes = np.arange(len(self.dataset))
            np.random.shuffle(dataset_indexes)
            self.ordering = np.array_split(dataset_indexes, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.cur_iter = 0
        self.iters_per_epoch = len(self.ordering)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.cur_iter < self.iters_per_epoch:
            batch_indexes = list(self.ordering[self.cur_iter])
            batch_samples = None
            batch_labels = None
            for i,batch_index in enumerate(batch_indexes):
                if i == 0:
                    batch_samples = self.dataset[batch_index][0]
                    batch_labels = [self.dataset[batch_index][1]]
                else:
                    batch_samples = np.concatenate((batch_samples, self.dataset[batch_index][0]), axis=0)
                    batch_labels += [self.dataset[batch_index][1]]
            self.cur_iter += 1
            return Tensor(batch_samples), Tensor(np.array(batch_labels))
        else:
            raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        self.images, self.labels = self.parse_mnist(self.image_filename, self.label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        batch = 1
        if isinstance(index, slice):
            batch = index.stop - index.start
        images = np.reshape(self.images[index], (batch, 28, 28, 1))
        aug_images = np.zeros((batch, 28, 28, 1))
        labels = None
        for i in range(batch):
            aug_image = self.apply_transforms(images[i])
            aug_images[i] = aug_image
        labels = self.labels[index]
        return aug_images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

    def parse_mnist(self, image_filename, label_filename):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                    data.  The dimensionality of the data should be 
                    (num_examples x input_dim) where 'input_dim' is the full 
                    dimension of the data, e.g., since MNIST images are 28x28, it 
                    will be 784.  Values should be of type np.float32, and the data 
                    should be normalized to have a minimum value of 0.0 and a 
                    maximum value of 1.0. The normalization should be applied uniformly
                    across the whole dataset, _not_ individual images.

                y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.uint8 and
                    for MNIST will contain the values 0-9.
        """
        ### BEGIN YOUR CODE
        with gzip.open(image_filename, 'r') as f:
            magic_number, image_count = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            X = np.frombuffer(f.read(), dtype=np.uint8)\
              .reshape((image_count, nrows*ncols))
            X = np.array(X / 255.0, dtype = np.float32)
        
        with gzip.open(label_filename, 'r') as f:
            magic_number, label_count = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)\
              .reshape((label_count))

        return (X, y)
        ### END YOUR CODE

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
