import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import PIL.Image as Image
from PIL import Image, ImageDraw, ImageFont
#from detr.models import detr as DETR

use_folders = [
    'bytes-cafe-2019-02-07_0',
    'clark-center-2019-02-28_0',
    'clark-center-intersection-2019-02-28_0',
    'cubberly-auditorium-2019-04-22_0',
    'forbes-cafe-2019-01-22_0',
    'gates-159-group-meeting-2019-04-03_0',
    'gates-basement-elevators-2019-01-17_1',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-basement-2019-01-25_0',
    'huang-lane-2019-02-12_0',
    'jordan-hall-2019-04-22_0',
    'memorial-court-2019-03-16_0',
    'packard-poster-session-2019-03-20_0',
    'packard-poster-session-2019-03-20_1',
    'packard-poster-session-2019-03-20_2',
    'stlc-111-2019-04-19_0',
    'svl-meeting-gates-2-2019-04-08_0',
    'svl-meeting-gates-2-2019-04-08_1',
    'tressider-2019-03-16_0'
]

image_address = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/images/image_stitched/"


class JRDBDataset(Dataset):
    def __init__(self, data_dir, use_folders=[], transforms=[]):
        paths = [os.path.join(data_dir, folder) for folder in use_folders]
        # files = [os.path.join(folder, x) for folder in paths for x in os.listdir(folder)]
        files = [folder + "/" + x for folder in paths for x in os.listdir(folder)]

        data_size = len(files)

        self.data_size = data_size
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_add = self.files[idx]
        image = Image.open(image_add)

        label_name = image_add[:-4].split("/")[-1]

        # image = image.astype(np.float32)

        if self.transforms:
            image = self.transforms(image)

        # print(image, label_name)
        return image, label_name


def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


#################################
#   Calculating mean, and std   #
#################################

# ds = JRDBDataset(data_dir=image_address, use_folders=use_folders, transforms=transforms.Compose([
# transforms.ToTensor()])) dataloader = DataLoader(dataset=ds, batch_size=64)
#
# mean, std = get_mean_std(dataloader)
#
# print(mean)
# print(std)

# mean: tensor([0.3691, 0.3965, 0.3752])
# std:  tensor([0.2716, 0.2871, 0.2799])