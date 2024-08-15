import os
import random
from functools import partial
from time import time
from typing import Optional, Union

import numpy as np
import torch
from anakin.artiboost import ArtiBoostLoader
from anakin.criterions.criterion import Criterion
from anakin.datasets.hodata import ho_collate
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.opt import arg, cfg
from anakin.opt_extra import data_generation_manager_parse
from anakin.utils import builder
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, TrainMode
from anakin.utils.netutils import build_optimizer, build_scheduler
from anakin.utils.recorder import Recorder
from anakin.utils.summarizer import Summarizer
from termcolor import colored

import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(0)


def _init_fn(worker_id):
    seed = int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def setup_seed(seed, conv_repeatable=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


def epoch_pass(
    train_mode: TrainMode,
    epoch_idx: int,
    data_loader: Optional[torch.utils.data.DataLoader],
    arch_model: Union[Arch, torch.nn.DataParallel],
    optimizer: Optional[torch.optim.Optimizer],
    criterion: Optional[Criterion],
    evaluator: Optional[Evaluator],
    summarizer: Optional[Summarizer],
    grad_clip: Optional[float] = None,
    use_grad_norm=True,
    model_raw=None
):
    if train_mode == TrainMode.TRAIN:
        arch_model.train()
    else:
        arch_model.eval()

    if evaluator:
        evaluator.reset_all()

    num_tasks = model_raw.get_num_tasks()
    weights = model_raw.get_task_weights()

    bar = etqdm(data_loader)
    for batch_idx, batch in enumerate(bar):
        predict_arch_dict = arch_model(batch)
        predicts = {}
        for key in predict_arch_dict.keys():
            predicts.update(predict_arch_dict[key])

        # ==== criterion >>>>
        if criterion:
            final_loss, losses, nan_loss_list, task_loss = criterion.compute_losses(predicts, batch, epoch_idx=epoch_idx)
        else:
            final_loss, losses = torch.Tensor([0.0]), {}
        # <<<<<<<<<<<<<<<<<<<<

        for nanloss in nan_loss_list:
            logger.info(f"warning: {nanloss} value is NaN in {epoch_idx}")

        # >>>> evaluate >>>>
        if evaluator:
            evaluator.feed_all(predicts, batch, losses)
        # <<<<<<<<<<<<<<<<<<

        # >>>> summarize >>>>
        if summarizer is not None and train_mode == TrainMode.TRAIN:
            summarizer.summarize_losses(losses)
        # <<<<<<<<<<<<<<<<<<

        
        # # >>>> backward >>>>
        if train_mode == TrainMode.TRAIN and not use_grad_norm:
            optimizer.zero_grad()  # for safety
            if len(nan_loss_list) == 0:
                final_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(arch_model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        # # <<<<<<<<<<<<<<<<<<
        if train_mode == TrainMode.TRAIN and use_grad_norm:
            last_shared_layer = model_raw.get_last_shared_layer()
            weighted_task_loss = torch.mul(task_loss, weights)
            initial_task_loss = task_loss.data.cpu().numpy()
            total_weighted_task_loss = weighted_task_loss.sum()
            optimizer.zero_grad()
            total_weighted_task_loss.backward(retain_graph=True)
            weights.grad.data = weights.grad.data * 0

            norms = []
            for i in range(num_tasks):
                gygw = torch.autograd.grad(task_loss[i], last_shared_layer.parameters(), retain_graph=True)
                norms.append(torch.norm(torch.mul(weights[i], gygw[0])))
            norms = torch.stack(norms)

            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            mean_norm = np.mean(norms.data.cpu().numpy())

            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 2), requires_grad=False).cuda()
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
            weights.grad = torch.autograd.grad(grad_norm_loss, weights)[0]
            optimizer.step()
        
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<
        bar_perfixes = {
            TrainMode.TRAIN: colored("train", "white", attrs=["bold"]),
            TrainMode.VAL: colored("val", "yellow", attrs=["bold"]),
            TrainMode.TEST: colored("test", "magenta", attrs=["bold"]),
        }
        bar.set_description(f"{bar_perfixes[train_mode]} Epoch {epoch_idx} | {str(evaluator)}")

    if use_grad_norm:
        normalize_coeff = num_tasks / torch.sum(weights.data, dim=0)
        weights.data = torch.mul(weights.data, normalize_coeff)
    
        with open("weights_record.txt", "a") as f:
            f.write(f"Epoch {epoch_idx}: {weights.data}\n")

def main_worker(time_f: float):
    recorder = Recorder(arg.exp_id, cfg, time_f=time_f)
    summarizer = Summarizer(arg.exp_id, cfg, tensorboard_path="./runs_new",time_f=time_f)
    logger.info(f"dump args: {arg, cfg['TRAIN']}")

    # region >>>>>>>>>>>>>>>>>>>> load test data >>>>>>>>>>>>>>>>>>>>
    test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=int(arg.workers),
        drop_last=False,
        collate_fn=ho_collate,
        worker_init_fn=_init_fn,
    )
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load model >>>>>>>>>>>>>>>>>>>>
    model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
    model = Arch(cfg, model_list=model_list)

    recorder.record_arch_graph(model)
    model = torch.nn.DataParallel(model).to(arg.device)

    optimizer = build_optimizer(
        model.module.models_params if hasattr(model, "module") else model.models_params,
        **cfg["TRAIN"],
    )
    scheduler = build_scheduler(optimizer, **cfg["TRAIN"])

    grad_clip = cfg["TRAIN"].get("GRAD_CLIP")
    if grad_clip is not None:
        logger.warning(f"Use gard clip norm {grad_clip}")
    
    use_grad_norm = cfg["TRAIN"].get("GRAD_NORM")

    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
        if arg.evaluate:
            cfg["TRAIN"]["EPOCH"] = epoch + 1  # enter into the train loop
    else:
        epoch = 0
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load criterion >>>>>>>>>>>>>>>>>>>>
    loss_list = builder.build_criterion_loss_list(cfg["CRITERION"],
                                                  preset_cfg=cfg["DATA_PRESET"],
                                                  LAMBDAS=cfg["LAMBDAS"])
    criterion = Criterion(cfg, loss_list=loss_list)
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load evaluator >>>>>>>>>>>>>>>>>>>>
    metrics_list = builder.build_evaluator_metric_list(cfg["EVALUATOR"], preset_cfg=cfg["DATA_PRESET"])
    evaluator = Evaluator(cfg, metrics_list=metrics_list)
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load artiboost >>>>>>>>>>>>>>>>>>>>
    # temp patch
    manager_dict_plus = {
        "VAL_FREQ": cfg["TRAIN"]["EVAL_FREQ"],
        "VAL_START_EPOCH": cfg["TRAIN"]["VAL_START_EPOCH"],
        "EPOCH": cfg["TRAIN"]["EPOCH"],
    }
    cfg["MANAGER"].update(manager_dict_plus)
    if not arg.evaluate:
        train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
        arg_extra = data_generation_manager_parse()

        # artiboost_loader = ArtiBoostLoader(
        #     train_data,
        #     arg=arg,
        #     arg_extra=arg_extra,
        #     cfg=cfg["MANAGER"],
        #     cfg_dataset=cfg["DATASET"],
        #     cfg_preset=cfg["DATA_PRESET"],
        #     time_f=time_f,
        #     batch_size=arg.batch_size,
        #     shuffle=True,
        #     num_workers=int(arg.workers),
        #     pin_memory=True,
        #     drop_last=arg.drop_last,
        #     collate_fn=ho_collate,
        #     random_seed=cfg["TRAIN"]["MANUAL_SEED"],
        # )
        artiboost_loader = torch.utils.data.DataLoader(
            dataset = train_data,
            batch_size = arg.batch_size,
            shuffle = True,
            num_workers = int(arg.workers),
            collate_fn = ho_collate,
            pin_memory = True,
            drop_last = arg.drop_last
        )

        if arg.resume:
            logger.warning(f"(experimental) resume artiboost loader, at epoch {epoch}")
            # recorder.resume_artiboost_loader(artiboost_loader, epoch, arg.resume)
    else:
        artiboost_loader = None
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Preprare train or test
    n_epoches = cfg["TRAIN"]["EPOCH"]

    # >>>>>>>>>>>>>>>>>>>> train >>>>>>>>>>>>>>>>>>>>
    logger.info(f"start training from {epoch} to {n_epoches}")
    for epoch_idx in range(epoch, n_epoches):
        if not arg.evaluate:
            # artiboost_loader.prepare()
            # * epoch pass >>>
            epoch_pass(
                train_mode=TrainMode.TRAIN,
                epoch_idx=epoch_idx,
                data_loader=artiboost_loader,
                arch_model=model,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=evaluator,
                summarizer=summarizer,
                grad_clip=grad_clip,
                use_grad_norm=use_grad_norm,
                model_raw=model_list[0]
            )
            scheduler.step()
            # artiboost_loader.step(epoch_idx=epoch_idx)
            # artiboost_loader.step_eval(epoch_idx=epoch_idx, evaluator=evaluator)
            # >>>>>>>>>>>>>>>>>>>> Save checkpoint >>>>>>>>>>>>>>>>>>>>
            recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
            recorder.record_evaluator(evaluator, epoch_idx, TrainMode.TRAIN)
            summarizer.summarize_evaluator(evaluator, epoch_idx, train_mode=TrainMode.TRAIN)
            # recorder.record_artiboost_loader(artiboost_loader, epoch_idx)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if epoch_idx % arg.test_freq == arg.test_freq - 1 or arg.evaluate:
            with torch.no_grad():
                epoch_pass(
                    train_mode=TrainMode.TEST,
                    epoch_idx=epoch_idx,
                    data_loader=test_loader,
                    arch_model=model,
                    optimizer=None,
                    criterion=criterion,
                    evaluator=evaluator,
                    summarizer=summarizer,
                    use_grad_norm=use_grad_norm,
                    model_raw=model_list[0]
                )
            recorder.record_evaluator(evaluator, epoch=epoch_idx, train_mode=TrainMode.TEST)
            summarizer.summarize_evaluator(evaluator, epoch_idx, train_mode=TrainMode.TEST)
            if arg.evaluate:
                break

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def main():
    exp_time = time()
    setup_seed(cfg["TRAIN"]["MANUAL_SEED"], cfg["TRAIN"].get("CONV_REPEATABLE", True))
    logger.info("====> Use Data Parallel <====")
    main_worker(exp_time)  # need to pass in renderer process group info


if __name__ == "__main__":
    main()
