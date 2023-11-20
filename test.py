import os
import torch
import argparse
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

def load_model(pretrained_path):
    # Load the model state dict from a given path
    state_dict = torch.load(pretrained_path, map_location="cpu")
    # Return the nested 'model' dict if it exists, else return the state dict
    return state_dict.get('model', state_dict)

if __name__ == "__main__":
    # Command-line arguments parsing
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load configuration files
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Setup output directory
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Logger setup
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Environment setup for CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Loading data and model
    dataloaders = make_dataloader(cfg)
    model = make_model(cfg, num_class=dataloaders['num_classes'], camera_num=dataloaders['camera_num'], view_num=dataloaders['view_num'])
    
    ckpt = load_model(cfg.TEST.WEIGHT)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    print('### missing keys=', missing_keys)
    print('### unexpected keys=', unexpected_keys)
    print('=========== success load pretrained model ===========')

    # Inference process
    device = "cuda"
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()
    
    get_memory = True    
    if get_memory:
        memory_global_cls = []
        memory_patch = []
        label = []
        img_path_list = []
        
        for n_iter, (img, pid, camid, camids, target_view, imgpath, occ_label) in enumerate(dataloaders['val_loader']):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                occ_label = occ_label.to(device)
                target = pid.to(device)

                memory_g_cls, memory_p, memory_label, memory_l_cls = model(img, target, cam_label=camids,
                                                                           view_label=target_view, get_memory=True, keep_rate=1.0)
                
                img_path_list.extend(imgpath) 
                memory_global_cls.append(memory_g_cls.cpu())
                memory_patch.append(memory_p.cpu())
                label.append(memory_label.cpu())
                # memory_local_cls.append(memory_l_cls.cpu())  # Removed as not used
        
        memory_global_cls = torch.cat(memory_global_cls, dim=0)
        memory_patch = torch.cat(memory_patch, dim=0)        
        label = torch.cat(label, dim=0)
        
        memory_global_cls = memory_global_cls[num_query:]
        gallery_memory_patch = memory_patch[num_query:]
        gallery_label = label[num_query:]
        img_path_list = img_path_list[num_query:]
        gallery_imgpath = img_path_list

        print('########################## make gallery memory done ##########################')
        print('memory_global_cls shape', memory_global_cls.shape)  # [num_gallery, C]
        print('gallery_memory_patch shape', gallery_memory_patch.shape)  # [num_gallery, N_s, C]
        print('gallery_label shape', gallery_label.shape)  # [num_gallery, ]
        print('gallery img_path_list shape', len(img_path_list)) 

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            rank_1, rank5 = do_inference(cfg,
                                         model,
                                         dataloaders['val_loader'],
                                         dataloaders['num_query'])
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 += rank_1
                all_rank_5 += rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
    else:
        do_inference(cfg,
                     model,
                     dataloaders['val_loader'],
                     dataloaders['num_query'], 
                     memory_global_cls,
                     None,
                     gallery_memory_patch,
                     gallery_label,
                     dataloaders['query_loader'],
                     dataloaders['gallery_loader'],
                     gallery_imgpath)
