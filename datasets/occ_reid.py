# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re
import urllib
import zipfile

import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset
import numpy as np


class OCC_REID(BaseImageDataset):
    """
    Occluded-reID
    Reference:

    Dataset statistics:
    # identities: 200 (train + query)
    # images:NO (train) + 1000 (query) + 1000 (gallery)
    # cameras: 
    """
    # train_dataset_dir = 'Occluded_Duke' # 用occ-duke的训练集
    train_dataset_dir = 'market1501' # 用market1501作为训练集
    dataset_dir = 'Occluded_REID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OCC_REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        # self.train_dir = osp.join(root, self.train_dataset_dir, 'bounding_box_train') # 用occ-duke的训练集
        self.train_dir = osp.join(root, self.train_dataset_dir, 'bounding_box_train') # 用market1501作为训练集
        
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')
        self.pid_begin = pid_begin
        self._download_data()
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, data_type='train')
        query = self._process_dir(self.query_dir, relabel=False, data_type='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False, data_type='gallery')

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-reID dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

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



    def _process_dir(self, dir_path, relabel=False, data_type=''):
        assert data_type == 'train' or data_type == 'query' or data_type == 'gallery'

        if data_type == 'train':
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # occ-reid *.jpg
            pattern = re.compile(r'([-\d]+)_c(\d)')
            pid_container = set()
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            dataset = []
            cam_container = set()

            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                assert 1 <= camid <= 8
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid))
                cam_container.add(camid)
            print(cam_container, 'cam_container')
            return dataset

        else:
            img_paths = glob.glob(osp.join(dir_path, '*', '*.tif'))  # ./Occluded_REID/occluded_body_images/001/001_01.tif

            pid_container = set()
            for img_path in img_paths:
                pid = int(img_path.split('/')[-2])
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            dataset = []
            cam_container = set()
            for img_path in img_paths:
                pid = int(img_path.split('/')[-2])
                if data_type == 'query':
                    camid = 1
                if data_type == 'gallery':
                    camid = 2
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid))
                cam_container.add(camid)
            print(cam_container, 'cam_container')
            return dataset
