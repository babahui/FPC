import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def do_inference(cfg, model, val_loader, num_query, memory_global_cls, 
                 gallery_memory_patch, query_loader, gallery_loader):
    """
    Perform inference for model evaluation.

    Args:
    cfg: Configuration object.
    model: The neural network model.
    val_loader: DataLoader for validation set.
    num_query: Number of query samples.
    memory_global_cls: Global class memory tensor.
    memory_local_cls: Local class memory tensor (not used in this function).
    gallery_memory_patch: Memory patch of gallery samples.
    query_loader: DataLoader for query set.
    gallery_loader: DataLoader for gallery set.

    Returns:
    Tuple: Rank-1 accuracy and Rank-5 accuracy.
    """
    start_time = time.time()  # Start timing
    device = "cuda"
    logger = logging.getLogger("fpc.test")
    logger.info("Enter inferencing")

    # Evaluator for re-identification metrics
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    # Setup model for inference
    if device:
        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # Process query images
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(query_loader):
        with torch.no_grad():
            img, camids, target_view, memory_global_cls, gallery_memory_patch, target = (
                img.to(device), camids.to(device), target_view.to(device),
                memory_global_cls.to(device), gallery_memory_patch.to(device),
                pid.to(device)
            )

            feat, idx = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                              memory_global_cls=memory_global_cls, memory_local_cls=None, 
                              gallery_memory_patch=gallery_memory_patch,
                              reconstruction=True, imgpath=imgpath, keep_rate=0.8)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # Process gallery images
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(gallery_loader):
        with torch.no_grad():
            img, camids, target_view, memory_global_cls, gallery_memory_patch, target = (
                img.to(device), camids.to(device), target_view.to(device),
                memory_global_cls.to(device), gallery_memory_patch.to(device),
                pid.to(device)
            )

            feat, idx = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                              memory_global_cls=memory_global_cls, memory_local_cls=None, 
                              gallery_memory_patch=gallery_memory_patch,
                              reconstruction=False, imgpath=imgpath, keep_rate=1.0)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # Compute metrics
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate total runtime
    logger.info("Inference completed in {:.2f} seconds".format(total_time))  # Output runtime

    return cmc[0], cmc[4]
