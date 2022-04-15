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


class VCClothes(object):
    """ VC-Clothes

    Reference:
        Wang et al. When Person Re-identification Meets Changing Clothes. In CVPR Workshop, 2020.

    URL: https://wanfb.github.io/dataset.html
    """
    dataset_dir = 'VC-Clothes'
    def __init__(self, root='data', mode='all', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        # 'all' for all cameras; 'sc' for cam2&3; 'cc' for cam3&4
        self.mode = mode 
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = self._process_dir_train()
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = self._process_dir_test()
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> VC-Clothes loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self):
        img_paths = glob.glob(osp.join(self.train_dir, '*.jpg'))
        img_paths.sort()
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes_id]
            dataset.append((img_path, pid, camid, clothes_id))
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self):
        query_img_paths = glob.glob(osp.join(self.query_dir, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(self.gallery_dir, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]:
                continue
            if self.mode == 'cc' and camid not in [3, 4]:
                continue
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]:
                continue
            if self.mode == 'cc' and camid not in [3, 4]:
                continue
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]:
                continue
            if self.mode == 'cc' and camid not in [3, 4]:
                continue
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            query_dataset.append((img_path, pid, camid, clothes_id))

        for img_path in gallery_img_paths:
            pid, camid, clothes, _ = pattern.search(img_path).groups()
            clothes_id = pid + clothes
            pid, camid = int(pid), int(camid)
            if self.mode == 'sc' and camid not in [2, 3]:
                continue
            if self.mode == 'cc' and camid not in [3, 4]:
                continue
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            gallery_dataset.append((img_path, pid, camid, clothes_id))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes


def VCClothesSameClothes(root='data', **kwargs):
    return VCClothes(root=root, mode='sc')


def VCClothesClothesChanging(root='data', **kwargs):
    return VCClothes(root=root, mode='cc')
