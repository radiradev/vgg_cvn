from dataloader import NeutDataset
from finetune import TransferLearningModel
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

def create_three_channels(image):
    return np.stack(arrays=(image, image, image), axis=2)


def valid_transform():
        return transforms.Compose(
            [
                transforms.Lambda(create_three_channels),
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


test_dataset = NeutDataset(
    images_path= '/data/rradev/cvn_data/raw_data',
    partition_path='/data/rradev/cvn/dataset/partition.p',
    labels_path='/data/rradev/cvn/dataset/labels.p',
    split='test',
    transform=valid_transform()
)

test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1)

images, labels = next(iter(test_dataloader))
model = TransferLearningModel()
output = model(images)
print(output)