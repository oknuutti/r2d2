# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import numpy as np
from PIL import Image

from .dataset import Dataset
from .pair_dataset import PairDataset, StillPairDataset


class AachenImages (Dataset):
    """ Loads all images from the Aachen Day-Night dataset 
    """
    def __init__(self, select='db day night', root='data/aachen', rgb=True, npy=False):
        Dataset.__init__(self)
        self.root = root
        self.img_dir = 'images_upright'
        self.select = set(select.split())
        assert self.select, 'Nothing was selected'
        
        self.imgs = []
        root = os.path.join(root, self.img_dir)
        for dirpath, _, filenames in os.walk(root):
            r = dirpath[len(root)+1:]
            if not(self.select & set(r.split('/'))): continue
            self.imgs += [os.path.join(r,f) for f in filenames if f.endswith('.npy' if npy else '.jpg')]
        
        self.nimg = len(self.imgs)
        assert self.nimg, 'Empty Aachen dataset'

    def get_key(self, idx):
        return self.imgs[idx]



class AachenImages_DB (AachenImages):
    """ Only database (db) images.
    """
    def __init__(self, **kw):
        AachenImages.__init__(self, select='db', **kw)
        self.db_image_idxs = {self.get_tag(i) : i for i,f in enumerate(self.imgs)}
    
    def get_tag(self, idx): 
        # returns image tag == img number (name)
        return os.path.split( self.imgs[idx][:-4] )[1]



class AachenPairs_StyleTransferDayNight (AachenImages_DB, StillPairDataset):
    """ synthetic day-night pairs of images 
        (night images obtained using autoamtic style transfer from web night images)
    """
    def __init__(self, root='data/aachen/style_transfer', **kw):
        StillPairDataset.__init__(self)
        AachenImages_DB.__init__(self, **kw)
        old_root = os.path.join(self.root, self.img_dir)
        self.root = os.path.commonprefix((old_root, root))
        self.img_dir = ''

        newpath = lambda folder, f: os.path.join(folder, f)[len(self.root):]
        self.imgs = [newpath(old_root, f) for f in self.imgs]

        self.image_pairs = []
        for fname in os.listdir(root):
            tag = fname.split('.jpg.st_')[0]
            self.image_pairs.append((self.db_image_idxs[tag], len(self.imgs)))
            self.imgs.append(newpath(root, fname))

        self.nimg = len(self.imgs)
        self.npairs = len(self.image_pairs)
        assert self.nimg and self.npairs



class AachenPairs_OpticalFlow (AachenImages_DB, PairDataset):
    """ Image pairs from Aachen db with optical flow.
    """
    def __init__(self, root='data/aachen', flow='optical_flow', rgb=True, npy=False, **kw):
        PairDataset.__init__(self)
        AachenImages_DB.__init__(self, root=root, rgb=rgb, npy=npy, **kw)
        self.root_flow = os.path.join(root, flow)
        self.npy = npy
        self.rgb = rgb

        # find out the subsest of valid pairs from the list of flow files
        flows = {f for f in os.listdir(os.path.join(self.root_flow, 'flow')) if f.endswith('.npy' if npy else '.png')}
        masks = {f for f in os.listdir(os.path.join(self.root_flow, 'mask')) if f.endswith('.npy' if npy else '.png')}
        assert flows == masks, 'Missing flow or mask pairs'
        
        make_pair = lambda f: tuple(self.db_image_idxs[v] for v in f[:-4].split('_'))
        self.image_pairs = [make_pair(f) for f in flows]
        self.npairs = len(self.image_pairs)
        assert self.nimg and self.npairs

    def get_mask_filename(self, pair_idx):
        tag_a, tag_b = map(self.get_tag, self.image_pairs[pair_idx])
        ext = 'npy' if self.npy else 'png'
        return os.path.join(self.root_flow, 'mask', f'{tag_a}_{tag_b}.{ext}')

    def get_mask(self, pair_idx):
        fname = self.get_mask_filename(pair_idx)
        if self.npy:
            return self.load_npy(fname)

        mask = np.asarray(Image.open(fname))
        if not self.npy and not isinstance(self.npy, bool):
            self.save_npy(fname[:-4] + '.npy', mask)

        return mask

    def get_flow_filename(self, pair_idx):
        tag_a, tag_b = map(self.get_tag, self.image_pairs[pair_idx])
        ext = 'npy' if self.npy else 'png'
        return os.path.join(self.root_flow, 'flow', f'{tag_a}_{tag_b}.{ext}')

    def get_flow(self, pair_idx):
        fname = self.get_flow_filename(pair_idx)
        try:
            if self.npy:
                return self.load_npy(fname)
            flow = self._png2flow(fname)
        except IOError:
            flow = open(fname[:-4], 'rb')
            help = np.fromfile(flow, np.float32, 1)
            assert help == 202021.25
            W, H = np.fromfile(flow, np.int32, 2)
            flow = np.fromfile(flow, np.float32).reshape((H, W, 2))
            flow = self._flow2png(flow, fname)

        if not self.npy and not isinstance(self.npy, bool):
            self.save_npy(fname[:-4] + '.npy', flow)

        return flow
    
    def get_pair(self, idx, output=()):
        if isinstance(output, str): 
            output = output.split()

        img1, img2 = map(self.get_image, self.image_pairs[idx])
        meta = {}
        
        if 'flow' in output or 'aflow' in output:
            flow = self.get_flow(idx)
            assert flow.shape[:2] == img1.size[::-1]
            meta['flow'] = flow
            H, W = flow.shape[:2]
            meta['aflow'] = flow + np.mgrid[:H,:W][::-1].transpose(1,2,0)
        
        if 'mask' in output:
            mask = self.get_mask(idx)
            assert mask.shape[:2] == img1.size[::-1]
            meta['mask'] = mask
        
        return img1, img2, meta




if __name__ == '__main__':
    print(aachen_db_images)
    print(aachen_style_transfer_pairs)
    print(aachen_flow_pairs)
    pdb.set_trace()
