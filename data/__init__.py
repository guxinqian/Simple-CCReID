import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from torch.utils.data import DataLoader
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.last import LaST
from data.datasets.ccvid import CCVID
from data.datasets.deepchange import DeepChange
from data.datasets.vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'vcclothes': VCClothes,
    'vcclothes_sc': VCClothesSameClothes,
    'vcclothes_cc': VCClothesClothesChanging,
    'last': LaST,
    'ccvid': CCVID,
    'deepchange': DeepChange,
}

VID_DATASET = ['ccvid']


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    if config.DATA.DATASET in VID_DATASET:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, 
                                                 sampling_step=config.DATA.SAMPLING_STEP,
                                                 seq_len=config.AUG.SEQ_LEN, 
                                                 stride=config.AUG.SAMPLING_STRIDE)
    else:
        dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_vid_transforms(config):
    spatial_transform_train = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.RandomHorizontalFlip(),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ST.RandomErasing(height=config.DATA.HEIGHT, width=config.DATA.WIDTH, probability=config.AUG.RE_PROB)
    ])
    spatial_transform_test = ST.Compose([
        ST.Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if config.AUG.TEMPORAL_SAMPLING_MODE == 'tsn':
        temporal_transform_train = TT.TemporalDivisionCrop(size=config.AUG.SEQ_LEN)
    elif config.AUG.TEMPORAL_SAMPLING_MODE == 'stride':
        temporal_transform_train = TT.TemporalRandomCrop(size=config.AUG.SEQ_LEN, 
                                                         stride=config.AUG.SAMPLING_STRIDE)
    else:
        raise KeyError("Invalid temporal sempling mode '{}'".format(config.AUG.TEMPORAL_SAMPLING_MODE))

    temporal_transform_test = None

    return spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    # video dataset
    if config.DATA.DATASET in VID_DATASET:
        spatial_transform_train, spatial_transform_test, temporal_transform_train, temporal_transform_test = build_vid_transforms(config)

        if config.DATA.DENSE_SAMPLING:
            train_sampler = DistributedRandomIdentitySampler(dataset.train_dense, 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # split each original training video into a series of short videos and sample one clip for each short video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train_dense, spatial_transform_train, temporal_transform_train),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        else:
            train_sampler = DistributedRandomIdentitySampler(dataset.train, 
                                                             num_instances=config.DATA.NUM_INSTANCES, 
                                                             seed=config.SEED)
            # sample one clip for each original training video during training
            trainloader = DataLoaderX(
                dataset=VideoDataset(dataset.train, spatial_transform_train, temporal_transform_train),
                sampler=train_sampler,
                batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                pin_memory=True, drop_last=True)
        
        # split each original test video into a series of clips and use the averaged feature of all clips as its representation
        queryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_query, spatial_transform_test, temporal_transform_test),
            sampler=DistributedInferenceSampler(dataset.recombined_query),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)
        galleryloader = DataLoaderX(
            dataset=VideoDataset(dataset.recombined_gallery, spatial_transform_test, temporal_transform_test),
            sampler=DistributedInferenceSampler(dataset.recombined_gallery),
            batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler
    # image dataset
    else:
        transform_train, transform_test = build_img_transforms(config)
        train_sampler = DistributedRandomIdentitySampler(dataset.train, 
                                                         num_instances=config.DATA.NUM_INSTANCES, 
                                                         seed=config.SEED)
        trainloader = DataLoaderX(dataset=ImageDataset(dataset.train, transform=transform_train),
                                 sampler=train_sampler,
                                 batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=True)

        galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                                   sampler=DistributedInferenceSampler(dataset.gallery),
                                   batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=True, drop_last=False, shuffle=False)

        if config.DATA.DATASET == 'prcc':
            queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                     sampler=DistributedInferenceSampler(dataset.query_same),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)
            queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                     sampler=DistributedInferenceSampler(dataset.query_diff),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
        else:
            queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test),
                                     sampler=DistributedInferenceSampler(dataset.query),
                                     batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                     pin_memory=True, drop_last=False, shuffle=False)

            return trainloader, queryloader, galleryloader, dataset, train_sampler

    

    
