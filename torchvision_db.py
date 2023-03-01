from __future__ import print_function
from os.path import join, isdir, sep, abspath, dirname, exists
from os import listdir, curdir, makedirs
import pandas as pd
from glob import glob
import numpy as np 
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Subset
from torchvision import datasets, transforms
#from RandAugment import RandAugment
try:
    import librosa
except:
    pass


def get_imagenet_db(root, trainf='train_org', testf='val', resize=None, crop=224, **kwargs):
    traindir = join(root, trainf, '')
    assert isdir(traindir)

    valdir = join(root, testf, '')
    assert isdir(valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = []
    if resize is not None:
        trans.append(transforms.Resize(resize))
    trans += [
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = datasets.ImageNet(root=traindir, split='train', transform=transforms.Compose(trans))
    rsz = resize if resize is not None else 256

    val_db = datasets.ImageNet(root=valdir, split='val', transform=transforms.Compose([
            transforms.Resize(rsz),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_db

def get_tiny_imagenet_db(root, trainf='train_org', testf='val', resize=None, crop=64, **kwargs):
    traindir = join(root, trainf, '')
    assert isdir(traindir)
    
    valdir = join(root, testf, '')
    assert isdir(valdir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = []
    if resize is not None:
        trans.append(transforms.Resize(resize))
    trans += [
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    print(trans)
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(trans))
    rsz = resize if resize is not None else 64
    val_db = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(rsz),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_db

def get_cub_db(root, trainf='train_org', testf='val', resize=150, crop=128, **kwargs):
    traindir = join(root, trainf, '')
    assert isdir(traindir)

    valdir = join(root, testf, '')
    assert isdir(valdir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = []
    if resize is not None:
        trans.append(transforms.Resize(resize))
    trans += [
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    print(trans)
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(trans))
    rsz = resize if resize is not None else 224
    val_db = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(rsz),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_db

def get_oxford_flower_db(root, trainf='train_org', testf='val', resize=150, crop=128, **kwargs):
    traindir = join(root, trainf, '')
    assert isdir(traindir)

    valdir = join(root, testf, '')
    assert isdir(valdir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = []
    if resize is not None:
        trans.append(transforms.Resize(resize))
    trans += [
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    print(trans)
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(trans))
    rsz = resize if resize is not None else 224
    val_db = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(rsz),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_db

def return_loaders(root, batch_size, test_bs=None, db='mnist', degrees=None, rsz=None,
                   augmentation=True, scale=None, shear=None, num_workers=2, n_reduce=None,**kwargs):
    """
    Return the loader for the data. This is used both for training and for
    validation.
    :param root: (str) Path of the root for finding the appropriate pkl/npy.
    :param batch_size: (int) The batch size for training.
    :param test_bs:  (int) The batch size for testing.
    :param db:  (str) The name of the database.
    :param kwargs:
    :return: The train and validation time loaders.
    """
    def _loader(db, batch_size, train=True, **kwargs):
        return DataLoader(db, shuffle=train, batch_size=batch_size,
                          num_workers=num_workers)

    if test_bs is None:
        test_bs = batch_size
    # # by default (i.e. with no normalization), the mnist, svhn,
    # # cifar10 in the range [0, 1].
    if 'cifar' in db:
        if augmentation:
            trans = [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip()]
            if degrees is not None:
                trans.append(transforms.RandomAffine(degrees, scale=scale, shear=shear))
            if rsz is not None:
                trans.append(transforms.Resize(rsz))
        else:
            trans = []
            if rsz is not None:
                trans.append(transforms.Resize(rsz))

        trans += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform_train = transforms.Compose(trans)
        # Add RandAugment with N, M(hyperparameter)
        #transform_train.transforms.insert(0, RandAugment(2, 0.5))
        print(transform_train)
        transform_test = transforms.Compose(trans[-2:])
        print(transform_test)
        if db == 'cifar100':
            trainset = datasets.CIFAR100(root=root, train=True,
                                         download=True, transform=transform_train)
            
            testset = datasets.CIFAR100(root=root, train=False,
                                        download=True, transform=transform_test)
        else:
            trainset = datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform_train)
            class_names = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            testset = datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform_test)
    elif db == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        trainset = datasets.SVHN(root=root, split='train',
                                 download=True, transform=transform)
        testset = datasets.SVHN(root=root, split='test',
                                download=True, transform=transform)
    elif db == 'imagenet':
        trainset, testset = get_imagenet_db(root, **kwargs)
        print('Loaded imagenet, with length {}.'.format(len(trainset)))
        print('Loaded imagenet, with valid set length {}.'.format(len(testset)))
    elif db == 'tiny_imagenet':
        trainset, testset = get_tiny_imagenet_db(root, **kwargs)
        print('Loaded imagenet, with length {}.'.format(len(trainset)))

        train_loader = _loader(trainset, batch_size, train=True, **kwargs)
        test_loader = _loader(testset, batch_size, train=False, **kwargs)
        return train_loader, test_loader
    elif db == 'MedMnist':
        trainset = MedMnist_specific(root, mode='train', **kwargs)
        testset = MedMnist_specific(root, mode='test', **kwargs)
    elif db == 'CUB':
        trainset, testset = get_cub_db(root, **kwargs)
        print('Loaded imagenet, with length {}.'.format(len(trainset)))

        train_loader = _loader(trainset, batch_size, train=True, **kwargs)
        test_loader = _loader(testset, batch_size, train=False, **kwargs)
        return train_loader, test_loader  
    elif db == 'oxford_flower':
        trainset, testset = get_oxford_flower_db(root, **kwargs)
        print('Loaded imagenet, with length {}.'.format(len(trainset)))

        train_loader = _loader(trainset, batch_size, train=True, **kwargs)
        test_loader = _loader(testset, batch_size, train=False, **kwargs)
        return train_loader, test_loader        
    elif db == 'STL10':
        if augmentation:
            trans = [transforms.RandomCrop(96, padding=4),
                     transforms.RandomHorizontalFlip()]
            if degrees is not None:
                trans.append(transforms.RandomAffine(degrees, scale=scale, shear=shear))
            if rsz is not None:
                trans.append(transforms.Resize(rsz))
        else:
            trans = []
            if rsz is not None:
                trans.append(transforms.Resize(rsz))
        trans += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_train = transforms.Compose(trans)

        transform_test = transforms.Compose(trans[-2:])
        
        trainset = datasets.STL10(root=root, split='train', download=True, transform=transform_train)
        testset = datasets.STL10(root=root, split='test', download=True, transform=transform_test)
        
        train_loader = _loader(trainset, batch_size, train=True, **kwargs)
        test_loader = _loader(testset, batch_size, train=False, **kwargs)
        return train_loader, test_loader

    else:
        raise NotImplementedError('db: {}'.format(db))
    if n_reduce is not None:
        print('Reducing the samples per class into {}.'.format(n_reduce))
        dt, cl = reduce_classes_dbset(trainset, n_reduce=n_reduce, permute=True)
        if db == 'svhn':
            trainset.data, trainset.labels = dt, cl
        else:
            trainset.data, trainset.targets = dt, cl
        print('[Debugging] Train samples: {}.\t\tTest samples: {}.'.format(len(trainset.data), len(testset.data)))

    if 'train' in kwargs.keys():
        del kwargs['train']
    
    # split the data set into train set and validation set
    # indices = list(range(len(trainset)))
    # split = int(np.floor(0.8 * len(trainset)))
    # train_indices = indices[:split]
    # val_indices = indices[split:]
    
    # train_sampler = SequentialSampler(train_indices)
    # valid_sampler = SequentialSampler(val_indices)
    
    # trainset_use = Subset(trainset, train_indices)
    # val_set = Subset(trainset, val_indices)
        
    # train_loader = _loader(trainset_use, batch_size, train=True, **kwargs)
    # val_loader = _loader(val_set, batch_size, train=False, **kwargs)

    train_loader = _loader(trainset, batch_size, train=True, **kwargs)
    val_loader = _loader(testset, test_bs, train=False, **kwargs)
    
    return train_loader, val_loader


def get_class_i(x, y, i, n_reduce=None):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    if n_reduce is not None:
        pos_i = pos_i[:n_reduce]
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


def reduce_classes_dbset(db_set, n_reduce=100, permute=True):
    """ Accepts a trainset torch db and reduces the samples for all classes in n_reduce. """
    db1 = db_set.data
    if not hasattr(db_set, 'targets'):
        db_set.targets = db_set.labels
    lbls = db_set.targets
    n_classes = int(np.max(db_set.targets)) + 1
    # # create the undersampled lists of data and samples.
    data, classes = [], []
    for cl_id in range(n_classes):
        samples = get_class_i(db1, lbls, cl_id, n_reduce=n_reduce)
        data.append(samples)
        classes.extend([cl_id] * n_reduce)
    # # convert into numpy arrays.
    data, classes =  np.concatenate(data, axis=0), np.array(classes, dtype=np.int)
    if permute:
        # # optionally permute the data to avoid having them sorted.
        permut1 = np.random.permutation(len(classes))
        data, classes = data[permut1], classes[permut1]
    return data, classes

def reduce_classes_dbset_longtailed(db_set, permute=True, lt_factor=None):
    """ Accepts a trainset torch db (which is assumed to have the same 
        number of samples in all classes) and creates a long-tailed 
        distribution with factor reduction factor. """
    db1 = db_set.data
    if not hasattr(db_set, 'targets'):
        db_set.targets = db_set.labels
    lbls = db_set.targets
    n_classes = int(np.max(db_set.targets)) + 1
    n_samples_class = int(db1.shape[0] // n_classes)
    # # create the undersampled lists of data and samples.
    data, classes = [], []
    for cl_id in range(n_classes):
        n_reduce = int(n_samples_class * lt_factor ** cl_id)
        samples = get_class_i(db1, lbls, cl_id, n_reduce=n_reduce)
        data.append(samples)
        classes.extend([cl_id] * n_reduce)
        print(n_reduce, len(data))
    # # convert into numpy arrays.
    data, classes =  np.concatenate(data, axis=0), np.array(classes, dtype=np.int)
    if permute:
        # # optionally permute the data to avoid having them sorted.
        permut1 = np.random.permutation(len(classes))
        data, classes = data[permut1], classes[permut1]
    return data, classes


class MedMnist_specific(Dataset):
    def __init__(self, root, number_db=0, mode='train', transform=None, **kwargs):
        # # https://medmnist.github.io/
        root = '/content/drive/My Drive/epfl_normalization_polynomials/datasets/medmnist/medmnist/medmnist'
        select_db = ['breastmnist', 'chestmnist',  'octmnist', 'pneumoniamnist',
                 'organmnist_axial', 'organmnist_coronal', 'organmnist_sagittal',
                 # # derma and pathmnist, retina have 3 channels; retina is not classification.
                 'dermamnist', 'pathmnist', 'retinamnist']
        self.root = root
        #self.path = '{}{}.npz'.format(root, select_db[number_db])
        self.path = '{}/{}.npz'.format(root, select_db[number_db])
        npz = np.load(self.path)
        # # load in self.data and convert unit8 to float32.
        self.data = npz['{}_images'.format(mode)].astype(np.float32) / 255.
        self.targets = npz['{}_labels'.format(mode)][:, 0].astype(np.int)
        if transform is None:
            transform = self._init_image_transform(**kwargs)
        self.transform = transform

    @staticmethod
    def _init_image_transform(size=0, **kwargs):
        trans = []
        if size > 0:
            trans.append(transforms.Resize((size, size)))
        trans.append(transforms.ToTensor())
        transform = transforms.Compose(trans)
        return transform

    def __getitem__(self, index):
        obj, label = self.data[index], self.targets[index]
        obj = self.transform(obj)
        return obj, label

    def __len__(self):
        return len(self.data)


def create_dataframe_emotion(root, **kwargs):
    """ Auxiliary function to get the metadata for emotion recognition db (RAVDESS)."""
    data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender',
                                    'intensity', 'statement', 'repetition', 'emotion'])
    count, ims = 0, []
    for i in sorted(listdir(root)):
        file_list = listdir(root + i)
        for f in file_list:
            nm = f.split('.')[0].split('-')
            path = root + i + sep + f
            src = int(nm[1])
            # # this is one-based by default.
            actor = int(nm[-1])
            # # convert the emotion (labels) to zero-based.
            emotion = int(nm[2]) - 1

            gender = 'female' if int(actor) % 2 == 0 else 'male'
            intensity = 0 if nm[3] == '01' else 1
            statement = 0 if nm[4] == '01' else 1
            repeat = 0 if nm[5] == '01' else 1

            data_df.loc[count] = [path, src, actor, gender, intensity, statement, repeat, emotion]
            count += 1
    return data_df


class EmotionRecognition(Dataset):
    def __init__(self, root, n_samples=None, transform=None, input_duration=3,
                 train=True, test=False, train_until=19, valid_until=22, 
                 n_mels=128, size_x=96, min_trim=15, max_trim=50, hop_length=512, 
                 n_fft=2048, augment=False, stretch_rate=None, noise=None, **kwargs):
        # # ensure librosa is installed.
        import librosa
        self.root = root
        self.is_train = train
        self.is_test = test
        self.input_duration = input_duration
        self.n_mels = n_mels
        self.size_x = size_x
        self.obj_list = create_dataframe_emotion(root, **kwargs)
        self.max_trim = max_trim
        self.min_trim = min_trim
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.augmentation = augment and train
        self.noise = noise
        self.stretch_rate = stretch_rate
        if stretch_rate is not None:
            self.strr = [min(stretch_rate, 1.), max(stretch_rate, 1.)]

        if n_samples is not None:
            self.obj_list = self.obj_list[:n_samples]
        if transform is None:
            transform = self._init_image_transform(**kwargs)
        self.transform = transform
        if self.is_train:
            # # crop the last actors for validation/testing.
            self.obj_list = self.obj_list[self.obj_list.actor < train_until]
        elif not self.is_test:
            # # crop for validation.
            self.obj_list = self.obj_list[self.obj_list.actor >= train_until][self.obj_list.actor < valid_until]
        else:
            print('Final testing mode.')
            self.obj_list = self.obj_list[self.obj_list.actor >= train_until] # valid_until]
        self.obj_list = self.obj_list.reset_index(drop=True)

    @staticmethod
    def _init_image_transform(size=0, **kwargs):
        trans = []
        if size > 0:
            trans.append(transforms.Resize((size, size)))
        trans.append(transforms.ToTensor())
        transform = transforms.Compose(trans)
        return transform

    def augment_data(self, data):
        if not self.augmentation:
            return data
        if self.noise is not None:
            noise_amp = self.noise * np.random.uniform() * np.amax(data)
            data = data + noise_amp * np.random.normal(size=data.shape[0])
        # # change dynamic value.
        dyn_change = np.random.uniform(low=1.0, high=1.25)
        data = data * dyn_change
        if self.stretch_rate is not None:
            stretch = np.random.uniform(low=self.strr[0], high=self.strr[1])
            data = librosa.effects.time_stretch(data, stretch)
        return data
    
    def _load_data(self, index):
        # # load the data.
        X, sample_rate = librosa.load(self.obj_list.path[index], res_type='kaiser_fast',
                                      duration=self.input_duration, sr=22050*2, offset=0.5)
        # # convert to spectrogram.
        sh = self.size_x - 10
        while sh < self.size_x:
            aa , _ = librosa.effects.trim(X, top_db=np.random.randint(self.min_trim, self.max_trim),
                                          frame_length=self.n_fft, hop_length=self.hop_length)
            aa = self.augment_data(aa)
            S = librosa.feature.melspectrogram(aa, sr=sample_rate, n_mels=self.n_mels, 
                                               hop_length=self.hop_length, n_fft=self.n_fft)
            sh = S.shape[1]
        log_S = librosa.power_to_db(S, ref=np.max)
        # # normalize in the [0, 1].
        log_S = (log_S + 80) / 80
        # # crop to ensure it's the same as the rest.
        crop_x = log_S.shape[1] - self.size_x
        if crop_x > 0:
            st = np.random.randint(0, crop_x)
            log_S = log_S[:, st: st + self.size_x]

        return log_S, self.obj_list.emotion[index]

    def __getitem__(self, index):
        obj, label = self._load_data(index)
        obj = self.transform(obj)
        return obj, label

    def __len__(self):
        return len(self.obj_list)


def wrapper_RAVDESS_list(root, train=True, test=False, **kwargs):
    """
    Thin wrapper that calls the EmotionRecognition class, i.e. it creates one
    instance for the training and one instance for the testing db (RAVDESS).
    :return: Two instances of the ObjectFilelist class (training and the validation).
    """
    trainset = EmotionRecognition(root, train=train, **kwargs)
    valid_db = EmotionRecognition(root, train=False, test=test, **kwargs)
    return trainset, valid_db





