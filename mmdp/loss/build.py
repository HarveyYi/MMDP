from mmdp.utils import Registry, check_availability

LOSS_REGISTRY = Registry("LOSS")


def build_loss(name, verbose=True, **kwargs):
    avai_loss = LOSS_REGISTRY.registered_names()
    check_availability(name, avai_loss)
    if verbose:
        print("Loss: {}".format(name))
    return LOSS_REGISTRY.get(name)(**kwargs)
