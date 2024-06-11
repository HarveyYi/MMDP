from mmdp.utils import Registry, check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_model(name, verbose=True, **kwargs):
    avai_models = MODEL_REGISTRY.registered_names()
    check_availability(name, avai_models)
    if verbose:
        print("Model name: {}".format(name))
    return MODEL_REGISTRY.get(name)(**kwargs)
