import random
import numpy as np


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        stride (int): Temporal sampling stride
    """

    def __init__(self, size=4, stride=8):
        self.size = size
        self.stride = stride

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            clips = []
            for i in range(self.size):
                    clips.append(frame_indices[len(frame_indices)//self.size*i : len(frame_indices)//self.size*(i+1)])
            out = []
            for i in range(self.size):
                out.append(random.choice(clips[i]))
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        stride (int): Temporal sampling stride
    """

    def __init__(self, size=8, stride=4):
        self.size = size
        self.stride = stride
        
    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size * self.stride:
            out = frame_indices[0 : self.size * self.stride : self.stride]
        else:
            out = frame_indices[0 : self.size]
            while len(out) < self.size:
                for index in out:
                    if len(out) >= self.size:
                        break
                    out.append(index)

        return out


class TemporalDivisionCrop(object):
    """Temporally crop the given frame indices by TSN.

    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size=4):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size:
            clips = []
            for i in range(self.size):
                clips.append(frame_indices[len(frame_indices)//self.size*i : len(frame_indices)//self.size*(i+1)])
            out = []
            for i in range(self.size):
                out.append(random.choice(clips[i]))
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]

        return out
