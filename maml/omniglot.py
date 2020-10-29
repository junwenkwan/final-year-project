import  torch.utils.data as data
import  os
import  os.path
import  errno


class Omniglot(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root #./omniglot
        self.transform = transform
        self.target_transform = target_transform


        self.all_items = find_classes(os.path.join(self.root))
        # ('0335_14.png', 'Futurama/character11', 
        # './omniglot/processed/images_background/Futurama/character11')

        # assigning each character to an index
        self.idx_classes = index_classes(self.all_items)
        # {'Sylheti/character25': 0, 'Sylheti/character03': 1, ... }

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        # 1527_15.png

        img = str.join('/', [self.all_items[index][2], filename])
        # ./omniglot/processed/images_evaluation/Sylheti/character25/1527_15.png

        target = self.idx_classes[self.all_items[index][1]]
        # self.all_items[index][1]: Sylheti/character25

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)


def find_classes(root_dir):
    # root_dir: ./omniglot/processed
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    # ('0335_14.png', 
    # 'Futurama/character11', 
    # './omniglot/processed/images_background/Futurama/character11')
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx
