import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))
