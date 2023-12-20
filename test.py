import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import copy
import torch


def load_model(pretrained_path, model):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        out_dict[k] = v
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("fpc", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, query_loader, gallery_loader = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    ckpt = load_model(cfg.TEST.WEIGHT, model)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    # print('### missing keys=', missing_keys)
    # print('### unexpected keys=', unexpected_keys)
    print('=========== success load pretrained model ===========')  


    device = "cuda"
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    model.eval()
    
    
    # Note: If the memory creation step takes a significant amount of time, consider using a feature storage 
    # approach to reduce model inference time in memory constuction. For a detailed discussion, refer to the appendix section of the paper.
    initialized = False
    start_idx = 0

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            target = pid.to(device)

            memory_g_cls, memory_p = model(img, target, cam_label=camids, view_label=target_view, get_memory=True, keep_rate=1.0)

            # Initialize tensors on the first iteration
            if not initialized:
                total_size = len(val_loader.dataset)
                dim1 = memory_g_cls.shape[1:]  # Get all dimensions except the first
                dim2 = memory_p.shape[1:]  # Get all dimensions except the first
                memory_global_cls = torch.zeros((total_size, *dim1))
                memory_patch = torch.zeros((total_size, *dim2))
                initialized = True

            end_idx = start_idx + img.size(0)
            memory_global_cls[start_idx:end_idx, ...] = memory_g_cls.cpu()
            memory_patch[start_idx:end_idx, ...] = memory_p.cpu()
            start_idx = end_idx

    # Split the data
    memory_global_cls = memory_global_cls[num_query:]
    gallery_memory_patch = memory_patch[num_query:]

    do_inference(cfg,
             model,
             val_loader,
             num_query, 
             memory_global_cls,
             gallery_memory_patch,
             query_loader,
             gallery_loader)

