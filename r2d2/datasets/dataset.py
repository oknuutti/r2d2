# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os
import json
import pdb
import numpy as np


class Dataset(object):
    ''' Base class for a dataset. To be overloaded.
    '''
    root = ''
    img_dir = ''
    nimg = 0

    def __len__(self):
        return self.nimg

    def get_key(self, img_idx):
        raise NotImplementedError()

    def get_filename(self, img_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_key(img_idx))

    @staticmethod
    def save_npy(fname, arr):
        with open(fname, 'wb') as fh:
            arr = np.save(fh, arr)
        return arr

    @staticmethod
    def load_npy(fname):
        with open(fname, 'rb') as fh:
            arr = np.load(fh)
        return arr

    def get_image(self, img_idx):
        fname = self.get_filename(img_idx)
        try:
            from PIL import Image
            if fname[-4:] == '.npy':
                arr = self.load_npy(fname)
                return Image.fromarray(arr)
            img = Image.open(fname).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fname, str(e)))

        rgb = getattr(self, 'rgb', True)
        npy = getattr(self, 'npy', False)
        if not npy and not isinstance(npy, bool):
            if not rgb:
                img = img.convert('L')
            self.save_npy(fname[:-4] + '.npy', np.array(img, dtype=np.uint8))

        return img


    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images' % self.nimg
        res += '\n  root: %s...\n' % self.root
        return res



class CatDataset (Dataset):
    ''' Concatenation of several datasets.
    '''
    def __init__(self, *datasets):
        assert len(datasets) >= 1
        self.datasets = datasets
        offsets = [0]
        for db in datasets:
            offsets.append(db.nimg)
        self.offsets = np.cumsum(offsets)
        self.nimg = self.offsets[-1]
        self.root = None

    def which(self, i):
        pos = np.searchsorted(self.offsets, i, side='right')-1
        assert pos < self.nimg, 'Bad image index %d >= %d' % (i, self.nimg)
        return pos, i - self.offsets[pos]

    def get_key(self, i):
        b, i = self.which(i)
        return self.datasets[b].get_key(i)

    def get_filename(self, i):
        b, i = self.which(i)
        return self.datasets[b].get_filename(i)

    def __repr__(self):
        fmt_str = "CatDataset("
        for db in self.datasets:
            fmt_str += str(db).replace("\n"," ") + ', '
        return fmt_str[:-2] + ')'




