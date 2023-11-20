import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def make_memory(data_loader, model, device, num_query):
    memory_global_cls = []
    memory_patch = []
    label = []
    memory_local_cls = []
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(data_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                occ_label = occ_label.to(device)
                target = pid.to(device)

                memory_g_cls, memory_p, memory_label, memory_l_cls = model(img, target, cam_label=camids, view_label=target_view, get_memory=True)

                memory_global_cls.append(memory_g_cls.cpu())
                memory_patch.append(memory_p.cpu())
                label.append(memory_label.cpu())
                memory_local_cls.append(memory_l_cls.cpu())

    memory_global_cls = torch.cat(memory_global_cls, dim=0)
    memory_patch = torch.cat(memory_patch, dim=0)        
    label = torch.cat(label, dim=0)
    memory_local_cls = torch.cat(memory_local_cls, dim=0)

    memory_global_cls = memory_global_cls[num_query:]
    gallery_memory_patch = memory_patch[num_query:]
    gallery_label = label[num_query:]
    memory_local_cls = memory_local_cls[num_query:]

    return memory_global_cls, gallery_memory_patch, gallery_label, memory_local_cls


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             gallery_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             local_rank,
             memory_global_cls,
             memory_local_cls,
             gallery_memory_patch,
             gallery_label,
             ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    mse_loss = torch.nn.MSELoss()
    
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, _, target_view, imgpath, occ_label) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # form gallery memory
            
#             memory_global_cls = memory_global_cls.to(device)
#             gallery_memory_patch = gallery_memory_patch.to(device)
#             gallery_label = gallery_label.to(device)
                
            with amp.autocast(enabled=True):
                score, feat, _, _ = model(img, target, cam_label=target_cam, view_label=target_view, get_memory=False, 
                                          memory_global_cls= memory_global_cls, gallery_memory_patch=gallery_memory_patch, 
                                          gallery_label=gallery_label, reconstruction=False, imgpath=imgpath)

                loss = loss_fn(score, feat, target, target_cam) 
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
                # acc = 0 # 这里要改

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

                
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    ''' both query and gallery do reconstrution '''
                    for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(val_loader): # query_loader
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            target = pid.to(device)
                            
                            feat = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                                         memory_global_cls=memory_global_cls, memory_local_cls=memory_local_cls, 
                                         gallery_memory_patch=gallery_memory_patch, gallery_label=gallery_label, 
                                         reconstruction=True, imgpath=imgpath, keep_rate=1.0)
                            evaluator.update((feat, pid, camid))

                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()

            else:
                model.eval()
                ''' both query and gallery do reconstrution '''
                
                # get memory each epoch
                memory_global_cls, gallery_memory_patch, gallery_label, memory_local_cls = make_memory(val_loader, model, device, num_query)
                memory_global_cls = memory_global_cls.to(device)   
                memory_local_cls = memory_local_cls.to(device)   
                gallery_memory_patch = gallery_memory_patch.to(device)
                gallery_label = gallery_label.to(device)
                        
                for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(val_loader): # _loader
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        target = pid.to(device)

                        feat = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                                     memory_global_cls=memory_global_cls, memory_local_cls=memory_local_cls, 
                                     gallery_memory_patch=gallery_memory_patch, gallery_label=gallery_label, 
                                     reconstruction=True, imgpath=imgpath, keep_rate=1.0)
                        evaluator.update((feat, pid, camid))
                        
                cmc, mAP, _, _, _, _, _ = evaluator.compute()                
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 memory_global_cls,
                 memory_local_cls,
                 gallery_memory_patch,
                 gallery_label,
                 query_loader,
                 gallery_loader,
                 gallery_imgpath):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # ''' #################### only query do reconstruction ##########################'''
    for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(query_loader): # query_loader
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # occ_label = occ_label.to(device)
            memory_global_cls = memory_global_cls.to(device)   
            memory_local_cls = None   
            gallery_memory_patch = gallery_memory_patch.to(device)
            gallery_label = gallery_label.to(device)
            target = pid.to(device)
            # imgpath = imgpath.to(device)
            
            feat, idx = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                         memory_global_cls=memory_global_cls, memory_local_cls=memory_local_cls, 
                         gallery_memory_patch=gallery_memory_patch, gallery_label=gallery_label, 
                         reconstruction=True, imgpath=imgpath, keep_rate=0.8, gallery_imgpath=gallery_imgpath) 
        
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            
    for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(gallery_loader): # gallery_loader
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            # occ_label = occ_label.to(device)
            memory_global_cls = memory_global_cls.to(device)   
            memory_local_cls = None  
            gallery_memory_patch = gallery_memory_patch.to(device)
            gallery_label = gallery_label.to(device)
            target = pid.to(device)
            # imgpath = imgpath.to(device)
            
            feat, idx = model(img, label=target, cam_label=camids, view_label=target_view, get_memory=False, 
                         memory_global_cls=memory_global_cls, memory_local_cls=memory_local_cls, 
                         gallery_memory_patch=gallery_memory_patch, gallery_label=gallery_label, 
                         reconstruction=False, imgpath=imgpath, keep_rate=1.0, gallery_imgpath=gallery_imgpath)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


