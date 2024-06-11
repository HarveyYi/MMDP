import os
import torch
import h5py
import numpy as np
import pandas as pd

import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from mmdp.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        fold=None,
        dataset_wrapper=None
    ):
        # Load dataset
        
        dataset = build_dataset(cfg, fold=fold)
        self.bins = dataset.bins
        self.omic_sizes = dataset.omic_sizes

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            # tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                # tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            # tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._classnames = dataset.classnames
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return ", ".join(map(str, self._classnames))

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME

        table = []
        table.append(["Dataset", dataset_name])
        table.append(["# fold", f"{self.dataset._fold}"])

        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# classnames", f"{self.classnames}"])
        table.append(
            ["# train", f"patients: {len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(
                ["# val", f"patients: {len(self.dataset.val):,}"])
        table.append(
            ["# test", f"patients: {len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, is_train=False):
        self.cfg = cfg
        self.data_source = data_source

        if is_train:
            self.sample = self.cfg.DATASET.PATH.SAMPLE
        else:
            self.sample = False

        self.num_patches = self.cfg.DATASET.PATH.NUM

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        # pathology featrues
        if self.cfg.DATASET.MODALITY == "omics":
            patch_features = torch.ones([1])
            mask = torch.ones([1])
        else:
            wsi_paths = item.histo_paths

            patch_features = []
            # load all slide_ids corresponding for the patient
            for wsi_path in wsi_paths:
                wsi_bag = torch.load(wsi_path)
                patch_features.append(wsi_bag)
            patch_features = torch.cat(patch_features, dim=0)

            if self.sample:
                max_patches = self.num_patches
                n_samples = min(patch_features.shape[0], max_patches)
                idx = np.sort(np.random.choice(
                    patch_features.shape[0], n_samples, replace=False))
                patch_features = patch_features[idx, :]

                # make a mask
                if n_samples == max_patches:
                    # sampled the max num patches, so keep all of them
                    mask = torch.zeros([max_patches])
                else:
                    # sampled fewer than max, so zero pad and add mask
                    original = patch_features.shape[0]
                    how_many_to_add = max_patches - original
                    zeros = torch.zeros(
                        [how_many_to_add, patch_features.shape[1]])
                    patch_features = torch.concat(
                        [patch_features, zeros], dim=0)
                    mask = torch.concat(
                        [torch.zeros([original]), torch.ones([how_many_to_add])])

            else:
                mask = torch.ones([1])

            patch_features = torch.tensor(np.array(patch_features)).float()
            mask = torch.tensor(np.array(mask))

        # omics data
        if self.cfg.DATASET.MODALITY == "pathology":
            omics_data = torch.ones([1])
        else:
            omics_data = item.omics_data

        # if omics_data[0].shape[0] == 0:
        #     print(f"missing genomic data for {item.histo_path}")

        if self.cfg.TASK.NAME == "Survival":
            #  {"labels": row["labels"], "survival_months": row["survival_months"], "censorship": row["censorship"]}
            label = torch.tensor(np.array(item.label["label"]))
            event_time = torch.tensor(np.array(item.label["event_time"]))
            censorship = torch.tensor(np.array(item.label["censorship"]))
            patient_id = item.patient_id
            if self.cfg.MODEL.NAME == "amisl":
                kmeans_path = os.path.join(
                    self.cfg.DATASET.CLUSTER_PATH, self.cfg.DATASET.NAME,  self.cfg.DATASET.PATH.FEATURE, f"{patient_id}.pt")
                prototype = torch.load(kmeans_path)
                prototype = prototype
                output = {
                    "label": label,
                    "event_time": event_time,
                    "censorship": censorship,
                    "patient_id": patient_id,
                    "histo_patch": patch_features,
                    "omics_data": omics_data,
                    "mask": mask,
                    "prototype": prototype,
                }

            else:
                output = {
                    "label": label,
                    "event_time": event_time,
                    "censorship": censorship,
                    "patient_id": patient_id,
                    "histo_patch": patch_features,
                    "omics_data": omics_data,
                    "mask": mask
                }

        else:
            if self.cfg.MODEL.NAME == "amisl":
                kmeans_path = item.histo_path.replace(
                    "wsi_data", "kmeans_label")
                prototype = torch.load(kmeans_path)
                prototype = torch.tensor(prototype)
                output = {
                    "label": label,
                    "patient_id": patient_id,
                    "histopath": patch_features,
                    "omics_data": omics_data,
                    "mask": mask,
                    "prototype": prototype,
                }
            else:
                output = {
                    "label": label,
                    "patient_id": patient_id,
                    "histopath": patch_features,
                    "omics_data": omics_data,
                    "mask": mask
                }

        return output

    def get_envent_and_cenorship(self):
        event_times, censorships = {}, {}
        for data in self.data_source:
            event_time, censorship = data.label['event_time'], data.label['censorship']
            patient_id = data.patient_id

            if patient_id in event_times:
                event_times[patient_id].append(event_time)
            else:
                event_times[patient_id] = [event_time]

            if patient_id in censorships:
                censorships[patient_id].append(censorship)
            else:
                censorships[patient_id] = [censorship]

        

        event_times = [np.mean(values) for _, values in event_times.items()]
        censorships = [int(np.mean(values))
                       for _, values in censorships.items()]

        return np.array(event_times), np.array(censorships)
