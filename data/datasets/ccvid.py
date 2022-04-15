import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json


class CCVID(object):
    """ CCVID

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.
    """
    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root, 'CCVID')
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()
 
        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes, _ = \
            self._process_data(self.train_path, relabel=True)
        clothes2label = self._clothes2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_clothes, _, _ = \
            self._process_data(self.query_path, relabel=False, clothes2label=clothes2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_clothes, _, _ = \
            self._process_data(self.gallery_path, relabel=False, clothes2label=clothes2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len, stride=stride)
       
        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs 
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_clothes = num_train_clothes + len(clothes2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets 

        logger = logging.getLogger('reid.dataset')
        logger.info("=> CCVID loaded")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------")
        logger.info("  subset       | # ids | # tracklets | # clothes")
        logger.info("  ---------------------------------------------")
        logger.info("  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_clothes))
        logger.info("  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_clothes))
        logger.info("  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_clothes))
        logger.info("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_clothes))
        logger.info("  ---------------------------------------------")
        logger.info("  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_clothes))
        logger.info("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        logger.info("  ---------------------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _clothes2label_test(self, query_path, gallery_path):
        pid_container = set()
        clothes_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        return clothes2label

    def _process_data(self, data_path, relabel=False, clothes2label=None):
        tracklet_path_list = []
        pid_container = set()
        clothes_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, clothes_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, clothes_label))
                clothes = '{}_{}'.format(pid, clothes_label)
                pid_container.add(pid)
                clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        if clothes2label is None:
            clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2clothes = np.zeros((num_pids, len(clothes2label)))

        for tracklet_path, pid, clothes_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*')) 
            img_paths.sort()

            clothes = '{}_{}'.format(pid, clothes_label)
            clothes_id = clothes2label[clothes]
            pid2clothes[pid2label[pid], clothes_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            session = tracklet_path.split('/')[0]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            else:
                camid = int(cam)

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, clothes_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, clothes_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, clothes_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx*sampling_step:], pid, camid, clothes_id))
                        else:
                            new_dataset.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid, clothes_id))
            else:
                new_dataset.append((img_paths, pid, camid, clothes_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, clothes_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, clothes_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths)//(seq_len*stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx : end_idx : stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # process the remaining sequence that can't be divisible by seq_len*stride        
            if len(img_paths)%(seq_len*stride) != 0:
                # reducing stride
                new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                    end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx : end_idx : new_stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, clothes_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert((vid2clip_index[idx, 1]-vid2clip_index[idx, 0]) == math.ceil(len(img_paths)/seq_len))

        return new_dataset, vid2clip_index.tolist()

