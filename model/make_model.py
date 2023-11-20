import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.emd import emd_similarity
from torch.nn.functional import normalize

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim) # [bs, 2, N/2, C]
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def patch_reconstruction(query_cls, query_patch, local_label, gallery_memory_cls, gallery_memory_patch, gallery_label, imgpath, gallery_imgpath, top_k=5):
    bs = query_cls.shape[0]
    rec_patch = []
    rec_cls = []
    
    for b in range(bs):                 
        ''' ##################################  1. use groudtruth label (up-bound of model) ################################## '''
        # serach_label = local_label[b]
        # neighbors_index = (gallery_label == serach_label).nonzero(as_tuple=True)[0]
        # random_index = torch.randperm(neighbors_index.size(0))
        # neighbors_index = neighbors_index[random_index]

        ''' ##################################  2. find k nearest neighbors  ##################################  '''
        # neighbors_index = []
        # global_neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), top_k)
        # neighbors_index = global_neighbors_index
        
        # cls_token dist
        # neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), 10, metric='cls_token')     
        
        #  Earth Mover's Distance (EMD)
        cls_neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), 100, imgpath=imgpath[b], metric='cls_token')    
        serach_patch = torch.transpose(query_patch[b], 0, 1) # [N, C] -> [C, N]
        gallery_patch = torch.transpose(gallery_memory_patch[cls_neighbors_index], 1, 2) # [100, N, C] -> [100, C, N]        
        emd_neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), 10, serach_patch, gallery_patch, 'both', 0.6, cls_neighbors_index, imgpath=imgpath[b])
        neighbors_index = cls_neighbors_index[emd_neighbors_index]

        ''' ##################################  final neighbors_index, then searching xxx neighbors ################################## '''
        # bs patch -> list
        neighbors_feat = gallery_memory_patch[neighbors_index] # [top_K, N_s, C]
        # neighbors_feat = torch.cat([torch.unsqueeze(query_patch[b], 0), neighbors_feat], dim=0) # cat query patch -> [top_K+1, N_s, C] 
        rec_patch.append(torch.unsqueeze(neighbors_feat, dim=0)) # list: bs* [1, top_k, N_s, C]

        # avg cls -> list
        neighbors_cls = gallery_memory_cls[neighbors_index] # [top_K, 1, C]
        neighbors_cls = torch.squeeze(neighbors_cls, 1) # [top_K, C]  # When dim is given, a squeeze operation is done only in the given dimension
        # neighbors_cls = torch.cat([torch.unsqueeze(query_cls[b], 0), neighbors_cls], dim=0) # cat query cls -> [top_K+1 ,C] 
        neighbors_cls = torch.mean(neighbors_cls, dim=0) # [C]
        rec_cls.append(torch.unsqueeze(neighbors_cls, dim=0)) # list: bs * [1, C]
                                
    # cat patch
    rec_patch = torch.cat(rec_patch, dim=0) # [bs, top_k, N_s, C]
    bs, _, _, C = rec_patch.shape
    rec_patch = torch.reshape(rec_patch, (bs, -1, C)) # [bs, top-K, N_s, C] -> [bs, top-K*N_s, C]
    
    # cat avg-cls
    rec_cls = torch.cat(rec_cls, dim=0) # [bs, C]
    rec_cls = torch.unsqueeze(rec_cls, dim=1) # [bs, 1, C]
            
    return rec_patch, rec_cls


def find_k_neighbors(serach_feat, gallery_feat, top_k, serach_patch=None, gallery_patch=None, metric=None, alpha=0.5, cls_neighbors_index=None, imgpath=None):
    # paras: 
    # serach_feat: [C], gallery_feat: [gallery_Num, C]
    # serach_local_feat: [K_part, C], gallery_local_feat: [gallery_Num, K_part, C]
    # serach_patch: [C, N], gallery_patch: [B, C, N]
    
    # serach_feat = torch.unsqueeze(serach_feat, dim=0) # serach_feat: [1, C]
    serach_feat = normalize(serach_feat, dim=0)
    gallery_feat = normalize(gallery_feat, dim=1)
    
    if metric == 'cls_token': 
        dist, _, _, _ = emd_similarity(None, serach_feat, None, gallery_feat, 0)  # cos-dist, 等价于l2-norm
        knn = dist.topk(top_k, largest=True)

    elif metric == 'patch_emd': 
        serach_patch = normalize(serach_patch, dim=0)
        gallery_patch = normalize(gallery_patch, dim=1)
        dist, _, _, _ = emd_similarity(serach_patch, serach_feat, gallery_patch, gallery_feat, 1, 'sc')
        knn = dist.topk(top_k, largest=True)

    elif metric == 'both':
        serach_patch = normalize(serach_patch, dim=0)
        gallery_patch = normalize(gallery_patch, dim=1)
        dist_emd, flows, _, _ = emd_similarity(serach_patch, serach_feat, gallery_patch, gallery_feat[cls_neighbors_index], 1, 'apc')
        dist_cls, _, _, _ = emd_similarity(None, serach_feat, None, gallery_feat[cls_neighbors_index], 0)  # 这里算cls-token的cos-dist
        dist = alpha * dist_cls + (1.0 - alpha) * dist_emd
        knn = dist.topk(top_k, largest=True)
        
    elif metric == 'Chamfer_Distance':
        dist_cls, _, _, _ = emd_similarity(None, serach_feat, None, gallery_feat[cls_neighbors_index], 0)  # 这里算cls-token的cos-dist
        dist_cd = Chamfer_Distance(serach_patch, gallery_patch)
        # print('###### Chamfer_Distance done ######')
        dist = alpha * dist_cls + (1.0 - alpha) * dist_cd
        knn = dist.topk(top_k, largest=False)
        
        knn.indices

    else:
        assert metric
        
    return knn.indices


def Chamfer_Distance(serach_patch, gallery_patch):
    # serach_patch: [N, C], gallery_patch: [K, N, C]    
    g_dist = []
    for g_k in gallery_patch:
        
        q_dist = []
        for q_i in serach_patch:
            dist = torch.norm(q_i-g_k, dim=1, p=2)  # l2-norm
            dist_min = torch.min(dist)
            dist_min = torch.tensor([dist_min]).to(dist_min.device)
            q_dist.append(dist_min)
        cat_q = torch.cat(q_dist, dim=0)
        g_k_dist = torch.sum(cat_q)
        g_k_dist = torch.tensor([g_k_dist]).to(g_k_dist.device)
        g_dist.append(g_k_dist)
        
    g_dist = torch.cat(g_dist)

    return g_dist

        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        
        # global branch
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        # reconstruction branch
        self.rec_b = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
                
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            # global feat
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            # reconstruction feat
            self.rec_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.rec_classifier.apply(weights_init_classifier)
        
        # global feat
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # reconstruction feat
        self.rec_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.rec_bottleneck.bias.requires_grad_(False)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None, get_memory=False,
                memory_global_cls=None, memory_local_cls=None, gallery_memory_patch=None, 
                gallery_label=None, reconstruction=False, imgpath=None, keep_rate=None, gallery_imgpath=None):

        # features = self.base(x, cam_label=cam_label, view_label=view_label)
        features, idxs = self.base(x, cam_label=cam_label, view_label=view_label, base_keep_rate=keep_rate)

        # global branch
        b1_feat = self.b1(features) # [64, N, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch/local branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        x = features[:, 1:]
                     
        feat = self.bottleneck(global_feat)
        

        ''' #################### get gallery memory ######################### '''
        if get_memory:
            global_cls = torch.squeeze(token)
            return features[:, 0:1], features[:, 1:], label, None

        ''' #################### reconstruction occluded patch features ######################### '''
        if reconstruction:
            top_k = 5
            rec_patch, rec_cls = patch_reconstruction(features[:, 0], features[:, 1:], label, memory_global_cls, 
                                                      gallery_memory_patch, gallery_label, imgpath, top_k, gallery_imgpath)
            
            rec_feat = self.rec_b(torch.cat((rec_cls, rec_patch), dim=1))
            rec_feat = rec_feat[:, 0]
            rec_feat_bn = self.rec_bottleneck(rec_feat)

            
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                # return None, None, occ_feat, occ_label # to-do changed

                cls_score = self.classifier(feat)
                if reconstruction:
                    cls_score_rec = self.rec_classifier(rec_feat_bn)
                return cls_score, global_feat, None, None 
            
        else:
            if self.neck_feat == 'after':
                # return feat # global feat
            
                if reconstruction:
                    return rec_feat_bn, idxs # rec feat
                else:
                    return feat, idxs
            
            else: # not pass 
                if reconstruction:
                    return rec_feat_bn, idxs # rec feat
                else:
                    return feat, idxs
            
            
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
