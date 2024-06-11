import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
from typing import Any, Dict, Optional

import gdown

from mmdp.utils import check_isfile



class Datum:
    """Data instance which defines the basic attributes.

    Args:
        histopath (str): histopothology path.
        omicpath (str): omics path.
        label (int): class label.
        classname (str): class name.
    """

    def __init__(self, patient_id, histo_paths, omics_data, classname, label:Optional[dict]=None):
        self._patient_id = patient_id
        self._histo_paths = histo_paths
        self._omics_data = omics_data
        self._label = label
        self._classname = classname

    @property
    def patient_id(self):
        return self._patient_id

    @property
    def histo_paths(self):
        return self._histo_paths
    
    @property
    def omics_data(self):
        return self._omics_data
    
    @property
    def label(self):
        return self._label
    
    @property
    def classname(self):
        return self._classname
    

    
    
class DatasetBase:
    """A unified dataset class for
    1) grading
    2) classing
    3) survival
    """

    dataset_dir = ""  # the directory where the dataset is stored


    def __init__(self, train=None, val=None, test=None):
        self._train = train  # labeled training data
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(self._train)
        self._lab2cname, self._classnames = self.get_lab2cname(self._train)

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(int(item.label["label"]))
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        # import pdb;pdb.set_trace()
        for item in data_source:
            container.add((item.label["label"], item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames


    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

