from mmdp.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg, **kwargs):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.TYPE, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.TYPE))
    return DATASET_REGISTRY.get(cfg.DATASET.TYPE)(cfg, **kwargs)