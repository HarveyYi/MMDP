import time
import random
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast


from mmsp.engine import TRAINER_REGISTRY, Trainer

from mmsp.utils import load_pretrained_weights, load_checkpoint, print_trainable_parameters
from mmsp.optim import build_optimizer, build_lr_scheduler
from mmsp.modeling import build_model
from mmsp.loss import build_loss

from mmsp.modeling import build_model

def split_chunk(data, target_length=512):
    # data = [bs, N, D]
    BS, N, D = data.shape
    num_splits = N //  target_length
    remainder = N % target_length

    if remainder > 0:
        padding_length = target_length - remainder
        padding = torch.zeros(BS, padding_length, D).to(data.device)
        data = torch.cat((data, padding), dim=1)  
        N += padding_length
        num_splits += 1  

    indices = torch.randperm(N, device=data.device)
    split_tensors = []
    
    for i in range(num_splits):
        start_index = i * target_length
        end_index = min((i + 1) * target_length, N)

        indices_in_range = indices[start_index:end_index]
        indices_in_range = torch.sort(indices_in_range).values

        sub_data = data[:, indices_in_range, :]
        split_tensors.append(sub_data)

    return split_tensors




@TRAINER_REGISTRY.register()
class MBTRAIN(Trainer):
    """

    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        omic_sizes = self.dm.dataset.omic_sizes
        self.use_bsm = cfg.DATASET.USE_BSM
        self.bs_micro = cfg.DATASET.BS_MICRO

        print("Building Model")
        print("Building model")
        num_classes = len(classnames)
        self.model = build_model(
            cfg.MODEL.NAME,
            verbose=cfg.VERBOSE,
            cfg=cfg,
            num_classes=num_classes,
            omic_sizes=omic_sizes,
        )

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            self.model.float()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print_trainable_parameters(self.model)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.loss_fn = build_loss(
            cfg.TASK.LOSS, alpha=cfg.LOSS.ALPHA, reduction=cfg.LOSS.REDUCTION)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        patient_id, x_path, x_mask, x_omic, label, event_time, censorship = self.parse_batch(
            batch)
        prec = self.cfg.TRAINER.PREC
        alpha = self.cfg.MODEL.UMEML.ALPHA
        if self.use_bsm:
            loss = 0.0
            cnt = 0
            # import pdb;pdb.set_trace()
            x_path_chunks = split_chunk(x_path, self.bs_micro)
            if prec == "amp":
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic}  
                    with autocast():
                        if self.cfg.TASK.NAME == "Survival":
                            logits, modular_loss_micro = self.model_inference(input)
                            loss_micro = self.loss_fn(
                                logits=logits, Y=label, c=censorship)
                        else:
                            logits, modular_loss_micro = self.model_inference(input)
                            loss_micro = self.loss_fn(logits, label)
                    loss += loss_micro + alpha * modular_loss_micro 
                    cnt+=1
                loss = loss / cnt    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
            else:
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic}  
                    if self.cfg.TASK.NAME == "Survival":
                        logits, modular_loss_micro = self.model_inference(input)
                        loss_micro = self.loss_fn(logits=logits,
                                            Y=label, c=censorship)
                    else:
                        logits, modular_loss_micro = self.model_inference(input)
                        loss_micro = self.loss_fn(logits, label)
                    loss += loss_micro + alpha * modular_loss_micro
                    cnt+=1
                loss = loss / cnt    
                self.model_backward_and_update(loss)  
        else:
            input = {"path": x_path, "omic": x_omic}  
            if prec == "amp":
                with autocast():
                    if self.cfg.TASK.NAME == "Survival":
                        logits = self.model_inference(input)
                        loss = self.loss_fn(
                            logits=logits, Y=label, c=censorship)
                    else:
                        logits = self.model_inference(input)
                        loss = self.loss_fn(logits, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
            else:
                if self.cfg.TASK.NAME == "Survival":
                    logits = self.model_inference(input)
                    loss = self.loss_fn(logits=logits,
                                        Y=label, c=censorship)
                else:
                    logits = self.model_inference(input)
                    loss = self.loss_fn(logits, label)
                self.model_backward_and_update(loss)
                

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        for _, batch in enumerate(tqdm(data_loader)):
            patient_id, x_path, x_mask, x_omic, label, event_time, censorship = self.parse_batch(
                batch)
            
            if self.use_bsm:
                all_logits = 0.
                all_S = 0.
                cnt = 0
                x_path_chunks = split_chunk(x_path, self.bs_micro)
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic}          
                    logits_micro = self.model_inference(input)
                    all_logits = all_logits + logits_micro
                    cnt+=1
                logits = all_logits / cnt
                
                if self.cfg.TASK.NAME == "Survival":

                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:

                    self.evaluator.process(patient_id, logits, label)
                        
            else:
                input = {"path": x_path, "mask": x_mask, "omic": x_omic}
                logits = self.model_inference(input)
                if self.cfg.TASK.NAME == "Survival":
                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:
                    self.evaluator.process(patient_id, logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
