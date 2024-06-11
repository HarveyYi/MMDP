from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1


# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"

# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bilinear"



###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = "./datasets_csv"
_C.DATASET.MODALITY = "multimodal"
_C.DATASET.NAME = "blca" 
_C.DATASET.TYPE = "Survival" 
_C.DATASET.FOLD = 0
_C.DATASET.SURVIVAL_ENDPOINT = "DSS"
_C.DATASET.USE_BSM = False
_C.DATASET.BS_MICRO = 512
_C.DATASET.SCPATH = False
_C.DATASET.CLUSTER_PATH = "./datasets/kmeans_label"

_C.DATASET.OMIC = CN()
_C.DATASET.OMIC.TYPE = "group"  # "group" or "pathway"
_C.DATASET.OMIC.PATHWAY = "combine"  # "combine" or "hallmarks" or "xena"
_C.DATASET.OMIC.DIM = 4999

_C.DATASET.PATH = CN()
_C.DATASET.PATH.FEATURE = "ctranspath"
_C.DATASET.PATH.SAMPLE = True
_C.DATASET.PATH.NUM = 4096
_C.DATASET.PATH.DIM = 1024



###########################
# Task
###########################
_C.TASK = CN()
_C.TASK.NAME = "Survival" # # 1. survival
_C.TASK.LOSS = "celoss" 
_C.TASK.INSTANCE_LOSS = "svmloss" 



###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# img0 denotes image tensor without augmentation
# Useful for consistency learning
_C.DATALOADER.RETURN_IMG0 = False
# Setting for the train_x data-loader
_C.DATALOADER.TRAIN = CN()
_C.DATALOADER.TRAIN.SAMPLER = "RandomSampler"
_C.DATALOADER.TRAIN.BATCH_SIZE = 1

# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
_C.DATALOADER.TEST.BATCH_SIZE = 1


###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ""
_C.MODEL.NAME = "abmil"
_C.MODEL.FUSION = None
_C.MODEL.SIZE = "small" # "small" or "big"
_C.MODEL.PATH1 = 8
_C.MODEL.PATH2 = 16
_C.MODEL.DROPOUT = 0.1
_C.MODEL.HIDDEN_DIM = 256
_C.MODEL.PROJECT_DIM = 256
_C.MODEL.OT_REG = 0.25
_C.MODEL.OT_TAU = 0.5
_C.MODEL.OT_IMPL = "pot-uot-l2"
_C.MODEL.CLAM_TYPE = "SB" # "SB" or "MB"
_C.MODEL.K_SAMPLE = 10
_C.MODEL.SUBTYPING = False
_C.MODEL.GATE = False
_C.MODEL.NUM_CLUSTERS = 10
_C.MODEL.NUM_HEADS = 1

_C.MODEL.UMEML = CN()
_C.MODEL.UMEML.PROTOTYPES = 5
_C.MODEL.UMEML.REGISTERS = 3
_C.MODEL.UMEML.ALPHA = 5.0


###########################
# LOSS
###########################
_C.LOSS = CN()
_C.LOSS.ALPHA = 0.5 
_C.LOSS.REDUCTION = "mean"
_C.LOSS.CMTA_ALPHA = 1.0
_C.LOSS.BAG_WEIGHT = 0.3

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adam"
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "single_step"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.OPTIM.WARMUP_EPOCH = -1
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "linear"
# Constant learning rate when type=constant
_C.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True



###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to only save the last one
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = "train"

###########################
# Test
###########################
_C.TEST = CN()

_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# Use test or val set for FINAL evaluation
_C.TEST.SPLIT = "test"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"


###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ""
_C.TRAINER.PREC = "fp32" # "fp32" "fp16" "amp"