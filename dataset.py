import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def random_crop(a, b, c, crop_size=128):

    H, W = a.shape[-2], a.shape[-1]
    assert H >= crop_size and W >= crop_size, "H and W must be >= crop_size"
    
    max_top = H - crop_size
    max_left = W - crop_size
    top = torch.randint(0, max_top + 1, size=(1,)).item()
    left = torch.randint(0, max_left + 1, size=(1,)).item()
    
    a_cropped = a[..., top:top+crop_size, left:left+crop_size]
    b_cropped = b[..., top:top+crop_size, left:left+crop_size]
    c_cropped = c[..., top:top+crop_size, left:left+crop_size]
    
    return a_cropped, b_cropped, c_cropped


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class fusion_dataset_gt(Dataset):
    def __init__(self, ir_path=None, vis_path=None, gt_path=None, model='train'):
        super(fusion_dataset_gt, self).__init__()
        self.filepath_vis, self.filenames_vis = prepare_data_path(vis_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
        self.filepath_gt, self.filenames_gt = prepare_data_path(gt_path)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt))
        if not (set(self.filenames_vis) == set(self.filenames_ir) == set(self.filenames_gt)):
            raise ValueError("可见光、红外和GT文件名集合不一致，请检查数据路径下的文件是否匹配！")
        self.model = model


    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        gt_path = self.filepath_gt[index]

        image_vis = np.array(Image.open(vis_path))
        if len(image_vis.shape) == 3:
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
        else:
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)

        image_inf = cv2.imread(ir_path, 0)
        if image_inf is not None:
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        else:
            image_ir = np.array(Image.open(ir_path).convert('L'), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)

        name = self.filenames_vis[index]

        image_gt = np.array(Image.open(gt_path))
        if len(image_gt.shape) == 3:
            image_gt = (
                    np.asarray(Image.fromarray(image_gt), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
        else:
            image_gt = np.asarray(Image.fromarray(image_gt), dtype=np.float32) / 255.0
            image_gt = np.expand_dims(image_gt, axis=0)

        if image_vis.shape[-1]==1024 and image_vis.shape[-2]==768 and self.model=='train':
            tensor_vis, tensor_ir, tensor_gt = random_crop(torch.tensor(image_vis, dtype=torch.float32), torch.tensor(image_ir,  dtype=torch.float32), torch.tensor(image_gt,  dtype=torch.float32),256)
            return(tensor_vis, tensor_ir, tensor_gt, name)
        else:
            return (    
                torch.tensor(image_vis, dtype=torch.float32),
                torch.tensor(image_ir, dtype=torch.float32),
                torch.tensor(image_gt, dtype=torch.float32),
                name,
            )

    def __len__(self):
        return self.length


class fusion_dataset(Dataset):
    def __init__(self, ir_path=None, vis_path=None):
        super(fusion_dataset, self).__init__()
        self.filepath_vis, self.filenames_vis = prepare_data_path(vis_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        image_vis = np.array(Image.open(vis_path))
        if len(image_vis.shape)==3:
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
        else:
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)


        image_inf = cv2.imread(ir_path, 0)
        if image_inf is not None:
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        else:
            image_ir = np.array(Image.open(ir_path).convert('L'), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)

        name = self.filenames_vis[index]

        return (    
            torch.tensor(image_vis, dtype=torch.float32),
            torch.tensor(image_ir, dtype=torch.float32),
            name,
        )   


    def __len__(self):
        return self.length