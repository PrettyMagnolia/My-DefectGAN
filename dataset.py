import os

from PIL import Image
from tqdm.auto import tqdm
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import show_tensor_images

defect_dict = {
    'Collision': 0,
    'Dirty': 1,
    'Gap': 2,
    'Scratch': 3
}


class DefectDataset(data.Dataset):
    """数据集类

    Args:
        dataset (list): 数据集
        transform (callable, optional): 数据预处理
    """

    def __init__(self, data_path, transform=None):
        super(DefectDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.images = []
        self.labels = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.jpg', 'png')):
                    img = Image.open(os.path.join(root, file))
                    self.images.append(self.transform(img))
                    self.labels.append(defect_dict[root.split('\\')[-1]])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


if __name__ == '__main__':
    dataset = DefectDataset(r'C:\Users\Bran.Liu\Desktop\Liu\code\Data_Quality_Toolkit\Annotation_contour')
    data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    for real, label in tqdm(data_loader):
        show_tensor_images(real)
        break
