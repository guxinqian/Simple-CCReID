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


class PRCC(object):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view
    """
    dataset_dir = 'prcc'
    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.val_dir = osp.join(self.dataset_dir, 'rgb/val')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        val, num_val_pids, num_val_imgs, num_val_clothes, _ = \
            self._process_dir_train(self.val_dir)

        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        logger = logging.getLogger('reid.dataset')
        logger.info("=> PRCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        logger.info("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        logger.info("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")

        self.train = train
        self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir_train(self, dir_path):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                dataset.append((img_dir, label, camid, clothes_id))
                pid2clothes[label, clothes_id] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes

    def _process_dir_test(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    # pid = pid2label[pid]
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        gallery_dataset.append((img_dir, pid, camid, clothes_id))
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        query_dataset_same_clothes.append((img_dir, pid, camid, clothes_id))
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        query_dataset_diff_clothes.append((img_dir, pid, camid, clothes_id))

        pid2imgidx = {}
        for idx, (img_dir, pid, camid, clothes_id) in enumerate(gallery_dataset):
            if pid not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)

        # get 10 gallery index to perform single-shot test
        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
                 
        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx
