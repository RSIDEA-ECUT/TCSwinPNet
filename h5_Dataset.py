import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class H5_TrainDataset_Up(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, idx):
        """
        gt:(19809, 4, 64, 64) float64
        lms:(19809, 4, 64, 64) float64
        pan:(19809, 1, 64, 64) float64
        """
        to_tensor = transforms.ToTensor()
        with h5py.File(self.data_path, 'r') as f:
            if "WV3" or "QuickBird" in self.data_path:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                lms = to_tensor((f.get('lms')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
            elif "GF1" in self.data_path:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                lms = to_tensor((f.get('lms')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
            else:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                lms = to_tensor((f.get('lms')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
            f.close()
            return lms, pan, gt

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            length = len(f.get('gt'))
            f.close()
            return length


class H5_TrainDataset_NoUp(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, idx):
        """
        gt:(19809, 4, 64, 64) float64
        ms:(19809, 4, 16, 16) float64
        pan:(19809, 1, 64, 64) float64
        """
        to_tensor = transforms.ToTensor()
        with h5py.File(self.data_path, 'r') as f:
            if "WV3" or "QuickBird" in self.data_path:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
            elif "GF1" in self.data_path:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
            else:
                gt = to_tensor((f.get('gt')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
            f.close()
            return ms, pan, gt

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            length = len(f.get('gt'))
            f.close()
            return length


class H5_FR_TestDataset_Up(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, idx):
        """
        lms:(20, 4, 512, 512) float64
        pan:(20, 1, 512, 512) float64
        """
        to_tensor = transforms.ToTensor()
        with h5py.File(self.data_path, 'r') as f:
            if "WV3" or "QuickBird" in self.data_path:
                lms = to_tensor((f.get('lms')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
            elif "GF1"  in self.data_path:
                lms = to_tensor((f.get('lms')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
            else:
                lms = to_tensor((f.get('lms')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                ms = to_tensor((f.get('ms')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
            f.close()
            return lms, ms, pan

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            length = len(f.get('lms'))
            f.close()
            return length


class H5_FR_TestDataset_NoUp(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, idx):
        """
        ms:(20, 4, 128, 128) float64
        pan:(20, 1, 512, 512) float64
        """
        to_tensor = transforms.ToTensor()
        with h5py.File(self.data_path, 'r') as f:
            if "WV3" or "QuickBird" in self.data_path:
                ms = to_tensor((f.get('ms')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 11 - 1)).transpose(1, 2, 0).astype(np.float32))
            elif "GF1"  in self.data_path:
                ms = to_tensor((f.get('ms')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 8 - 1)).transpose(1, 2, 0).astype(np.float32))
            else:
                ms = to_tensor((f.get('ms')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
                pan = to_tensor((f.get('pan')[idx] / (2 ** 10 - 1)).transpose(1, 2, 0).astype(np.float32))
            f.close()
            return ms, pan

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            length = len(f.get('ms'))
            f.close()
            return length


if __name__ == '__main__':
    data_path = r"D:\LZC\PanCollection\GF2\reduced-test\show_gf2_example.h5"
    dataset = H5_TrainDataset_NoUp(data_path)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(train_dataloader):
        lms, pan, gt = data
        print(lms.shape, pan.shape, gt.shape)
        break

