from torch.utils import data
from imageio import imread
import numpy as np

class ImageDataset(data.Dataset):
    
    def __init__(self,filenames,labels,tfms=None,label_normalize=True):
        self.filenames=filenames
        self.labels=labels
        self.transforms=tfms
        self.label_normalize = label_normalize
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,index):
        file = self.filenames[index]
        sample = imread(file)
        ratio = np.asarray([224/490,224/326])
        if self.transforms:
            sample = self.transforms(sample)
#             self.labels = self.labels*ratio
#         sample = sample.permute(2,1,0)
        y = self.labels[index]
        if self.label_normalize:
            y=y*224
        return sample,y

class ImageDataset_hflip(data.Dataset):
    
    def __init__(self,filenames,labels,tfms=None,label_normalize=True):
        self.filenames=filenames
        self.labels=labels
        self.transforms=tfms
        self.label_normalize = label_normalize
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,index):
        file = self.filenames[index]
        sample = imread(file)
        sample = sample[:, ::-1, :] #Horizontal flip
        if self.transforms:
            sample = self.transforms(sample)
        
        label=self.labels[index]
        y = np.asarray([1-label[0],label[1]],dtype=np.float32)

        if self.label_normalize:
            y=y*224
        return sample,y

class ImageDataset_vflip(data.Dataset):
    
    def __init__(self,filenames,labels,tfms=None,label_normalize=True):
        self.filenames=filenames
        self.labels=labels
        self.transforms=tfms
        self.label_normalize = label_normalize
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,index):
        file = self.filenames[index]
        sample = imread(file)
        sample = sample[::-1, :, :] #Vertical flip
        if self.transforms:
            sample = self.transforms(sample)
        
        label=self.labels[index]
        y = np.asarray([label[0],1-label[1]],dtype=np.float32)
        if self.label_normalize:
            y=y*224
        return sample,y
