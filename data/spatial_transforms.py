import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class RandomCrop(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, size, p=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.height, self.width = self.size
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if not self.cropping:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(self.tl_x * x_maxrange))
        y1 = int(round(self.tl_y * y_maxrange))
        return resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))

    def randomize_parameters(self):
        self.cropping = random.uniform(0, 1) < self.p
        self.tl_x = random.random()
        self.tl_y = random.random()


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.

    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.
        
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, height=256, width=128, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.height = height
        self.width = width
       
    def __call__(self, img):
        if self.re:
            return img

        if img.size()[0] == 3:
            img[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
            img[1, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[1]
            img[2, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[2]
        else:
            img[0, self.x1:self.x1+self.h, self.y1:self.y1+self.w] = self.mean[0]
        return img

    def randomize_parameters(self):
        self.re = random.uniform(0, 1) < self.probability
        self.h, self.w, self.x1, self.y1 = 0, 0, 0, 0
        whether_re = False
        if self.re:
            for attempt in range(100):
                area = self.height*self.width

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                self.h = int(round(math.sqrt(target_area * aspect_ratio)))
                self.w = int(round(math.sqrt(target_area / aspect_ratio)))
                if self.w < self.width and self.h < self.height:
                    self.x1 = random.randint(0, self.height - self.h)
                    self.y1 = random.randint(0, self.width - self.w)
                    whether_re = True
                    break

        self.re = whether_re