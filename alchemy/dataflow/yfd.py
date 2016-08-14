import os
import random
from collections import defaultdict

from alchemy.dataflow.dataset import Dataset

class YFD(Dataset):

    def __init__(self, aligned_face_dir):
        self.db_name = 'Youtube Face DB'
        self.aligned_face_dir = aligned_face_dir

    @classmethod
    def aligned_train_val(self, data_dir, shuffle=True, num_train_val=None,
                          num_trains=None, train_ratio=None):
        assert num_trains or train_ratio

        self.aligned_face_dir = data_dir
        id_images = defaultdict(list)
        class_id = 0
        for d in os.listdir(data_dir):
            if d.startswith('.'):
                continue
            fd = os.path.join(data_dir, d)

            for root, dirs, filenames in os.walk(fd):
                for fn in filenames:
                    fullname = os.path.join(root, fn)
                    id_images[class_id].append(fullname)
            class_id += 1

        train_val_set = {}
        for cid, images in id_images.items():
            if shuffle:
                random.shuffle(images)
            if num_train_val:
                images = images[:num_train_val]

            num_trains = num_trains if num_trains else \
                    int(len(images) * train_ratio)

            train_val_set[cid] = {
                'train': images[:num_trains],
                'val': images[num_trains:]
            }

        return train_val_set

    def __str__(self):
        return self.db_name
