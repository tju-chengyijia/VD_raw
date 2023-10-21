import numpy as np
import torch
import argparse
import cv2, os, glob
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile


class data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER
        self.frames_each_video = args.frames_each_video
        file_type = image_list[0].split('.')[-1]
        self.file_type = '.%s' % file_type

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        index = index 
        image_in_gt = self.image_list[index]
        video_number = image_in_gt.split('/')[-1][0:5]#0:5
        number = int(image_in_gt.split('/')[-1][5:7])#5:7
        image_in = video_number + '%05d' % number + self.file_type

        assert self.args.NUM_AUX_FRAMES > 0
        
        if self.mode == 'train':
            path_srcs = []
            path_tars = []
            # path_refs = []
            path_src_auxs = []

            if self.args.use_temporal:
                # currently, this code can only use 2 output frames/ 2 branches
                for k in range(2):
                    if number > (self.frames_each_video - 2):
                        number = self.frames_each_video - 2
                    number_cur = number + k
                    path_tar = self.args.TRAIN_DATASET + '/gt_rgb/' + video_number + '%02d' % number_cur + '.png'#self.file_type
                    path_src = self.args.TRAIN_DATASET + '/moire_raw/' + video_number + '%02d' % number_cur + '.npz'#self.file_type
                    # path_ref = self.args.TRAIN_DATASET + '/moire_rgb/' + video_number + '%02d' % number_cur + '.png'#self.file_type

                    path_srcs.append(path_src)
                    path_tars.append(path_tar)
                    # path_refs.append(path_ref)

                    for i in range(1, self.args.NUM_AUX_FRAMES + 1):#取出number_cur的前一帧和后一帧，如果没有那就直接取number_cur本身
                        if i % 2 == 0:
                            number_tmp = number_cur + i // 2 * self.args.FRAME_INTERVAL
                        else:
                            number_tmp = number_cur - (i + 1) // 2 * self.args.FRAME_INTERVAL
                            if number_tmp < 0:
                                number_tmp = number_cur
                        #aux_in = video_number + '%02d' % number_tmp + self.file_type
                        path_src_aux = self.args.TRAIN_DATASET + '/moire_raw/' + video_number + '%02d' % number_tmp + '.npz'#aux_in
                        if not os.path.isfile(path_src_aux):
                            path_src_aux = path_src
                        if self.args.MODE == 'single':
                            path_src_aux = path_src
                        path_src_auxs.append(path_src_aux)

            # elif self.args.use_flow:
            #     # currently, this code can only use 2 output frames/ 2 branches
            #     path_flows = []
            #     path_masks = []
            #     for k in range(2):
            #         if number > (self.frames_each_video - 2):
            #             number = self.frames_each_video - 2
            #         number_cur = number + k
            #         path_tar = self.args.TRAIN_DATASET + '/target/' + video_number + '%05d' % number_cur + self.file_type
            #         path_src = self.args.TRAIN_DATASET + '/source/' + video_number + '%05d' % number_cur + self.file_type
            #         # flow between two frames (branch 1 and branch 2)
            #         path_flow = self.args.flow_path + video_number + '%05d' % number_cur + '.npz'
            #         path_mask = self.args.flow_path + video_number + '%05d' % number_cur + '.png'
            #
            #         if not os.path.isfile(path_flow):
            #             path_flow = self.args.flow_path + video_number + '%05d' % number + '.npz'
            #         if not os.path.isfile(path_mask):
            #             path_mask = self.args.flow_path + video_number + '%05d' % number + '.png'
            #
            #         path_srcs.append(path_src)
            #         path_tars.append(path_tar)
            #         path_flows.append(path_flow)
            #         path_masks.append(path_mask)
            #
            #         for i in range(1, self.args.NUM_AUX_FRAMES + 1):
            #             if i % 2 == 0:
            #                 number_tmp = number_cur + i // 2 * self.args.FRAME_INTERVAL
            #             else:
            #                 number_tmp = number_cur - (i + 1) // 2 * self.args.FRAME_INTERVAL
            #                 if number_tmp < 0:
            #                     number_tmp = number_cur
            #             aux_in = video_number + '%05d' % number_tmp + self.frames_each_video
            #             path_aux = self.args.TRAIN_DATASET + '/source/' + aux_in
            #             if not os.path.isfile(path_aux):
            #                 path_aux = path_src
            #             if self.args.MODE == 'single':
            #                 path_aux = path_src
            #             path_src_auxs.append(path_aux)

            else:  
                if number > (self.frames_each_video - 2):
                    number = self.frames_each_video - 2
                number_cur = number
                path_tar = self.args.TRAIN_DATASET + '/gt_rgb/' + video_number + '%02d' % number_cur + '.png'#self.file_type
                path_src = self.args.TRAIN_DATASET + '/moire_raw/' + video_number + '%02d' % number_cur + '.npz'#self.file_type
                # path_ref = self.args.TRAIN_DATASET + '/moire_rgb/' + video_number + '%02d' % number_cur + '.png'#self.file_type

                path_srcs.append(path_src)
                path_tars.append(path_tar)
                # path_refs.append(path_ref)

                for i in range(1, self.args.NUM_AUX_FRAMES + 1):#取出number_cur的前一帧和后一帧，如果没有那就直接取number_cur本身
                    if i % 2 == 0:
                        number_tmp = number_cur + i // 2 * self.args.FRAME_INTERVAL
                    else:
                        number_tmp = number_cur - (i + 1) // 2 * self.args.FRAME_INTERVAL
                        if number_tmp < 0:
                            number_tmp = number_cur
                    #aux_in = video_number + '%02d' % number_tmp + self.file_type
                    path_src_aux = self.args.TRAIN_DATASET + '/moire_raw/' + video_number + '%02d' % number_tmp + '.npz'#aux_in
                    if not os.path.isfile(path_src_aux):
                        path_src_aux = path_src
                    if self.args.MODE == 'single':
                        path_src_aux = path_src
                    path_src_auxs.append(path_src_aux)
                    
                
            if self.loader == 'crop':
                x = random.randint(0, self.args.WIDTH//2 - self.args.CROP_SIZE//2)
                y = random.randint(0, self.args.HEIGHT//2 - self.args.CROP_SIZE//2)
                labels = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_tars,'rgb')
                # refs = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_refs,'rgb')
                moire_imgs = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_srcs,'raw')
                moire_imgs_aux = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_src_auxs,'raw')
                # if self.args.use_flow:
                #     flows = crop_loader_flow(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_flows)
                #     masks = crop_loader_mask(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_masks)

            elif self.loader == 'reszie':
                labels = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_tars)
                moire_imgs = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_srcs)
                moire_imgs_aux = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_src_auxs)

            elif self.loader == 'default':
                labels = default_loader(path_tars)
                # refs = default_loader(path_refs)
                moire_imgs = default_loader(path_srcs)
                moire_imgs_aux = default_loader(path_src_auxs)
                
            #data augmentation
            # len_labels,len_refs,len_mimgs,len_mimgsaux = len(labels),len(refs),len(moire_imgs),len(moire_imgs_aux)
            # imgs = labels+refs+moire_imgs+moire_imgs_aux
            # imgs = augment(imgs)
            # labels = imgs[:len_labels]
            # refs = imgs[len_labels:len_labels+len_refs]
            # moire_imgs = imgs[len_labels+len_refs:len_labels+len_refs+len_mimgs]
            # moire_imgs_aux = imgs[len_labels+len_refs+len_mimgs:]
            # import matplotlib.pyplot as plt
            # for i in imgs:
            #     if i.shape[0]==4:
            #         plt.imshow(i[[0,1,3],:,:].permute(1,2,0).cpu().numpy())
            #     else:
            #         plt.imshow(i.permute(1,2,0).cpu().numpy())
            #     plt.show()

        elif self.mode == 'val':
            path_tar = self.args.TEST_DATASET + '/gt_rgb/' + video_number + '%02d' % number + '.png'#image_in_gt
            path_src = self.args.TEST_DATASET + '/moire_raw/' + video_number + '%02d' % number + '.npz'#image_in
            # path_ref = self.args.TEST_DATASET + '/moire_rgb/' + video_number + '%02d' % number + '.png'

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                #aux_in = video_number + '%02d' % number_tmp + self.file_type
                path_src_aux = self.args.TEST_DATASET + '/moire_raw/' + video_number + '%02d' % number_tmp + '.npz'#aux_in
                if not os.path.isfile(path_src_aux):
                    path_src_aux = path_src
                if self.args.MODE == 'single':
                    path_src_aux = path_src
                path_src_auxs.append(path_src_aux)

            # if self.loader == 'crop':
            #     x = random.randint(0, self.args.WIDTH//2 - self.args.CROP_SIZE//2)
            #     y = random.randint(0, self.args.HEIGHT//2 - self.args.CROP_SIZE//2)
            #     labels = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, [path_tar],'rgb')
            #     refs = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, [path_ref],'rgb')
            #     moire_imgs = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, [path_src],'raw')
            #     moire_imgs_aux = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_src_auxs,'raw')
            # elif self.loader == 'reszie':
            #     labels = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_tar])
            #     moire_imgs = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_src])
            #     moire_imgs_aux = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_src_auxs)
            # elif self.loader == 'default':
            #     labels = default_loader([path_tar])
            #     refs = default_loader([path_ref])
            #     moire_imgs = default_loader([path_src])
            #     moire_imgs_aux = default_loader(path_src_auxs)
            
            labels = default_loader([path_tar],'rgb')
            # refs = default_loader([path_ref],'rgb')
            moire_imgs = default_loader([path_src],'raw')
            moire_imgs_aux = default_loader(path_src_auxs,'raw')

        elif self.mode == 'test':
            path_tar = self.args.TEST_DATASET + '/gt_rgb/' + video_number + '%02d' % number + '.png'#image_in_gt
            path_src = self.args.TEST_DATASET + '/moire_raw/' + video_number + '%02d' % number + '.npz'#image_in
            # path_ref = self.args.TEST_DATASET + '/moire_rgb/' + video_number + '%02d' % number + '.png'

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                #aux_in = video_number + '%02d' % number_tmp + self.file_type
                path_src_aux = self.args.TEST_DATASET + '/moire_raw/' + video_number + '%02d' % number_tmp + '.npz'#aux_in
                if not os.path.isfile(path_src_aux):
                    path_src_aux = path_src
                if self.args.MODE == 'single':
                    path_src_aux = path_src
                path_src_auxs.append(path_src_aux)

            labels = default_loader([path_tar],'rgb')
            # refs = default_loader([path_ref],'rgb')
            moire_imgs = default_loader([path_src],'raw')
            moire_imgs_aux = default_loader(path_src_auxs,'raw')

        else:
            print('Unrecognized mode! Please select either "train" or "val" or "test"')
            raise NotImplementedError

        data['number'] = video_number + '%05d' % number
        data['in_img'] = moire_imgs
        data['in_img_aux'] = moire_imgs_aux
        data['label'] = labels
        # data['ref'] = refs

        #if not self.mode == 'test':
        #    data['label'] = labels

        # if self.mode == 'train' and self.args.use_flow:
        #     data['flow'] = flows
        #     data['mask'] = masks

        return data

    def __len__(self):
        return len(self.image_list)



def augment(imgs, hflip=True, rotation=True, cflip=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5
    cflip = cflip and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            #cv2.flip(img, 1, img)
            torch.flip(img,dims=[1])
        if vflip:  # vertical
            #cv2.flip(img, 0, img)
            torch.flip(img,dims=[2])
        if rot90:
            img = torch.transpose(img, 1, 2)
        if cflip:
            torch.flip(img,dims=[0])
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs


# def load_raw(path):
#     raw_img = np.load(path)
#     raw_data = raw_img['data'].transpose((2,0,1))
#     bl = raw_img['black_level_per_channel'][0]
#     wl = raw_img['white_level']
#     norm_factor = wl - bl
#     raw_data = (raw_data- bl)/norm_factor
#     raw_data = raw_data.astype(np.float32)

#     return raw_data

def load_raw(path):
    raw_img = np.load(path)
    raw_data = raw_img['data'].transpose((2,0,1))
    bl = raw_img['black_level_per_channel'][0]
    wl = raw_img['white_level']
    norm_factor = wl - bl
    raw_data = (raw_data- bl)/norm_factor
    raw_data = raw_data.astype(np.float32)
    # add camera_whitebalance
    cwb = raw_img['camera_whitebalance']
    cwb_rggb = np.expand_dims(np.expand_dims(np.array([cwb[0],cwb[1],cwb[1],cwb[2]]), axis=1), axis=2)
    raw_data = raw_data*cwb_rggb
    raw_data = raw_data.astype(np.float32)

    return raw_data


def default_loader(path_set,type_f):
    imgs = []
    for path in path_set:
        if type_f=='rgb':
            img = Image.open(path).convert('RGB')
            img = default_toTensor(img)
        elif type_f=='raw':
            img = load_raw(path)
            img = torch.from_numpy(img)
        imgs.append(img)

    return imgs


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def crop_loader(crop_size_x, crop_size_y, x, y, path_set, type_f, pad_size=100, pad=False):
    imgs = []
    for path in path_set:
        if type_f=='rgb':
            img = Image.open(path).convert('RGB')
            img = img.crop((2*x, 2*y, 2*x + crop_size_x, 2*y + crop_size_y))
            img = default_toTensor(img)
        elif type_f=='raw':
            img = load_raw(path)
            img = img[:,y:y + crop_size_y//2,x:x + crop_size_x//2]
            img = torch.from_numpy(img)
        '''
        if pad:
            img = add_margin(img, pad_size, pad_size, pad_size, pad_size, (123, 117, 104))
        '''
        imgs.append(img)

    return imgs


def crop_loader_mask(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.crop((x, y, x + crop_size_x, y + crop_size_y))
        img = 1 - default_toTensor(img)
        imgs.append(img)

    return imgs
    
    
def crop_loader_flow(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = np.load(path)['flow']
        img = img[y:(y+crop_size_y), x:(x+crop_size_x), :]
        img = default_toTensor(img)
        imgs.append(img)
        
    return imgs


def resize_loader(resize_size_h, resize_size_w, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize_size_w, resize_size_h), Image.BICUBIC)
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)
