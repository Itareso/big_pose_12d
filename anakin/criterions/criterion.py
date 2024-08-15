from typing import Dict, Tuple, List

import torch
from anakin.utils.logger import logger
from anakin.utils.misc import camel_to_snake
import math


class TensorLoss(object):
    def __init__(self):
        super(TensorLoss, self).__init__()
        self.output_key = f"{camel_to_snake(type(self).__name__)}_output"

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        target_device = None
        losses = {}

        # Get device
        for key in preds.keys():
            if isinstance(preds[key], torch.Tensor):
                target_device = preds[key].device
                break
        if target_device is None:
            logger.error("Cannot found valid Tensor with device")
            raise RuntimeError()

        final_loss = torch.Tensor([0.0]).float().to(target_device)
        return final_loss, losses


class Criterion(TensorLoss):
    def __init__(self, cfg: Dict, loss_list: List[TensorLoss]) -> None:
        super(Criterion, self).__init__()
        self._loss_list = loss_list
        self._loss_lambdas = {}
        self._cfg = cfg

        # parse lambdas
        lambdas = list(cfg["LAMBDAS"])
        for i in range(len(loss_list)):
            lambda_ = lambdas[i]
            self._loss_lambdas[type(loss_list[i]).__name__] = lambda_

        # logging
        logger.info(f"CONSTRUCT CRITERION WITH LAMBDAS: ")
        for loss in loss_list:
            name = type(loss).__name__
            logger.info(f"  |  LAMBDA_{name} : {self._loss_lambdas[name]}")

    @property
    def loss_list(self) -> List[TensorLoss]:
        return self._loss_list

    @property
    def loss_lambdas(self) -> Dict[str, float]:
        return self._loss_lambdas

    def compute_losses(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        weighted_loss_sum, total_losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        task_loss = torch.empty(0).to(weighted_loss_sum.device)
        flag = False
        sum_lambda = 0
        sum_lambda_epoch = 0
        for loss in self.loss_list:
            name = type(loss).__name__
            loss_lambda = self.loss_lambdas[name]
            sum_lambda += loss_lambda
            # if name == "AccConsistencyLoss" or name == "BetaConsistencyLoss":
            #     loss_lambda *= kwargs["epoch_idx"]/100
            sum_lambda_epoch += loss_lambda
        ratio = sum_lambda / sum_lambda_epoch
        nan_loss_list = []
        for loss in self.loss_list:
            name = type(loss).__name__
            final_loss, losses = loss(preds, targs, **kwargs)
            if math.isnan(final_loss):
                nan_loss_list.append(name)
                continue
            # if name == "AccConsistencyLoss" or name == "BetaConsistencyLoss":
            #     weighted_loss_sum += self.loss_lambdas[name] * final_loss * ratio * kwargs["epoch_idx"]/100
            # else:
            #     weighted_loss_sum += self.loss_lambdas[name] * final_loss * ratio
            weighted_loss_sum += self.loss_lambdas[name] * final_loss
            total_losses.update(losses)
            task_loss = torch.cat((task_loss, final_loss.view(1)))

        assert "final_loss" not in total_losses, "unexpected premature final loss encountered"
        total_losses["final_loss"] = weighted_loss_sum
        return weighted_loss_sum, total_losses, nan_loss_list, task_loss
