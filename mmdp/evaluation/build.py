from mmdp.utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def build_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TASK.NAME, avai_evaluators)
    if cfg.VERBOSE:
        print("Loading evaluator: {}".format(cfg.TASK.NAME))
    return EVALUATOR_REGISTRY.get(cfg.TASK.NAME)(cfg, **kwargs)
