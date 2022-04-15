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


class LaST(object):
    """ LaST

    Reference:
        Shu et al. Large-Scale Spatio-Temporal Person Re-identification: Algorithm and Benchmark. arXiv:2105.15076, 2021.

    URL: https://github.com/shuxjweb/last

    Note that LaST does not provide the clothes label for val and test set.
    """
    dataset_dir = "last"
    def __init__(self, root='data', **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.test_query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        self._check_before_run()

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(self.train_dir)

        train, num_train_pids = self._process_dir(self.train_dir, pid2label=pid2label, clothes2label=clothes2label, relabel=True)
        val_query, num_val_query_pids = self._process_dir(self.val_query_dir, relabel=False)
        val_gallery, num_val_gallery_pids = self._process_dir(self.val_gallery_dir, relabel=False, recam=len(val_query))
        test_query, num_test_query_pids = self._process_dir(self.test_query_dir, relabel=False)
        test_gallery, num_test_gallery_pids = self._process_dir(self.test_gallery_dir, relabel=False, recam=len(test_query))

        num_total_pids = num_train_pids+num_val_gallery_pids+num_test_gallery_pids
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> LaST loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset        | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train         | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), len(clothes2label)))
        logger.info("  query(val)    | {:5d} | {:8d} |".format(num_val_query_pids, len(val_query)))
        logger.info("  gallery(val)  | {:5d} | {:8d} |".format(num_val_gallery_pids, len(val_gallery)))
        logger.info("  query         | {:5d} | {:8d} |".format(num_test_query_pids, len(test_query)))
        logger.info("  gallery       | {:5d} | {:8d} |".format(num_test_gallery_pids, len(test_gallery)))
        logger.info("  --------------------------------------------")
        logger.info("  total         | {:5d} | {:8d} | ".format(num_total_pids, num_total_imgs))
        logger.info("  --------------------------------------------")

        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = len(clothes2label)
        self.pid2clothes = pid2clothes

    def get_pid2label_and_clothes2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))            # [103367,]
        img_paths.sort()

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            clothes_container.add(clothes)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.val_gallery_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))

    def _process_dir(self, dir_path, pid2label=None, clothes2label=None, relabel=False, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        img_paths.sort()
        
        dataset = []
        pid_container = set()
        for ii, img_path in enumerate(img_paths):
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            if relabel and clothes2label is not None:
                clothes_id = clothes2label[clothes]
            else:
                clothes_id = pid
            dataset.append((img_path, pid, camid, clothes_id))
        num_pids = len(pid_container)

        return dataset, num_pids