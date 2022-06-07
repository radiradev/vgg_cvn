import torch
import zlib
import pickle
import numpy as np


class NeutDataset(torch.utils.data.Dataset):
    outputs = [
        "anti",
        "flavor",
        "interaction",
        "protons",
        "pions",
        "pizeros",
        "neutrons",
    ]
    num_outputs = 7

    def __init__(
        self,
        images_path='/afs/cern.ch/work/r/rradev/public/vgg_cvn/data',
        partition_path="dataset/partition.p",
        labels_path="dataset/labels.p",
        cells=500,
        planes=500,
        views=3,
        split='train',
        transform = None
    ):
        self.cells = cells
        self.planes = planes
        self.views = views
        self.images_path = images_path
        self.transform = transform
        
        with open(labels_path, "rb") as l_file:
            self.labels = pickle.load(l_file)

        with open(partition_path, "rb") as p_file:
           self.partition = pickle.load(p_file)
        self.list_IDs = self.partition[split]
        

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        with open(
            self.images_path
            + "/"
            + ID.split(".")[0].lstrip("a")
            + "/images/"
            + ID
            + ".gz",
            "rb",
        ) as image_file:
            image = torch.tensor(
                np.fromstring(
                    zlib.decompress(image_file.read()), dtype=np.uint8, sep=""
                ).reshape(self.views, self.planes, self.cells)
            )[0] # only collection view
        if self.transform:
            image = self.transform(image)

        # return labels:
        labels = np.array(self.labels[ID])
        labels[labels > 3] = 3
        
        return image, labels

