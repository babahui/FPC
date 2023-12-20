import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from utils.emd import emd_similarity
from torch.nn.functional import normalize


def patch_reconstruction(query_cls, query_patch, gallery_memory_cls, gallery_memory_patch, top_k=10):
    """
    Reconstructs patch and class features for query samples based on their nearest neighbors in the gallery.
    
    Args:
    query_cls (Tensor): Query class features.
    query_patch (Tensor): Query patch features.
    gallery_memory_cls (Tensor): Class features of gallery samples.
    gallery_memory_patch (Tensor): Patch features of gallery samples.
    top_k (int): Number of nearest neighbors to consider.

    Returns:
    Tuple[Tensor, Tensor]: Reconstructed patch and class features for the query samples.
    """

    batch_size = query_cls.shape[0]
    reconstructed_patch = []
    reconstructed_cls = []
    
    for b in range(batch_size):
        # Using Earth Mover's Distance (EMD) to find nearest neighbors
        cls_neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), 100, metric='cls_token')    
        transposed_query_patch = torch.transpose(query_patch[b], 0, 1)  # Transpose the query patch
        transposed_gallery_patch = torch.transpose(gallery_memory_patch[cls_neighbors_index], 1, 2)  # Transpose the gallery patch
        emd_neighbors_index = find_k_neighbors(query_cls[b], torch.squeeze(gallery_memory_cls), top_k, transposed_query_patch, 
                                               transposed_gallery_patch, 'both', 0.6, cls_neighbors_index)
        neighbors_index = cls_neighbors_index[emd_neighbors_index]

        # Getting the patch features of the neighbors
        neighbors_feature = gallery_memory_patch[neighbors_index]
        reconstructed_patch.append(torch.unsqueeze(neighbors_feature, dim=0))

        # Calculating the average class feature of the neighbors
        neighbors_class = torch.squeeze(gallery_memory_cls[neighbors_index], 1)
        mean_neighbors_class = torch.mean(neighbors_class, dim=0)
        reconstructed_cls.append(torch.unsqueeze(mean_neighbors_class, dim=0))
    
    # Here we input tokens of gallery neighbors for model inference.
    # Concatenating patch features
    reconstructed_patch = torch.cat(reconstructed_patch, dim=0)
    batch_size, _, _, C = reconstructed_patch.shape
    reconstructed_patch = torch.reshape(reconstructed_patch, (batch_size, -1, C))
    
    # Concatenating class features
    reconstructed_cls = torch.cat(reconstructed_cls, dim=0)
    reconstructed_cls = torch.unsqueeze(reconstructed_cls, dim=1)
            
    return reconstructed_patch, reconstructed_cls


def find_k_neighbors(search_feat, gallery_feat, top_k, search_patch=None, gallery_patch=None, metric=None, alpha=0.5, cls_neighbors_index=None):
    """
    Find k nearest neighbors in the gallery for a given search feature.

    Args:
    search_feat (Tensor): Feature vector of the search sample [C].
    gallery_feat (Tensor): Feature matrix of the gallery samples [gallery_Num, C].
    top_k (int): Number of nearest neighbors to find.
    search_patch (Tensor, optional): Patch feature of the search sample [C, N].
    gallery_patch (Tensor, optional): Patch feature matrix of the gallery samples [B, C, N].
    metric (str, optional): The metric used for finding neighbors ('cls_token', 'patch_emd', 'both', 'Chamfer_Distance').
    alpha (float, optional): Weighting factor for combining different metrics.
    cls_neighbors_index (Tensor, optional): Preselected neighbor indices for some metrics.

    Returns:
    Tensor: Indices of the k nearest neighbors.
    """

    # Normalize the search and gallery features
    search_feat = normalize(search_feat, dim=0)
    gallery_feat = normalize(gallery_feat, dim=1)

    # Calculate distance based on the specified metric
    if metric == 'cls_token':
        # Cosine distance, equivalent to l2-norm
        dist, _, _, _ = emd_similarity(None, search_feat, None, gallery_feat, 0)
        knn = dist.topk(top_k, largest=True)

    elif metric == 'patch_emd':
        # Normalize patch features
        search_patch = normalize(search_patch, dim=0)
        gallery_patch = normalize(gallery_patch, dim=1)
        dist, _, _, _ = emd_similarity(search_patch, search_feat, gallery_patch, gallery_feat, 1, 'sc')
        knn = dist.topk(top_k, largest=True)

    elif metric == 'both':
        # Combine cls_token and patch_emd metrics
        search_patch = normalize(search_patch, dim=0)
        gallery_patch = normalize(gallery_patch, dim=1)
        dist_emd, flows, _, _ = emd_similarity(search_patch, search_feat, gallery_patch, gallery_feat[cls_neighbors_index], 1, 'apc')
        dist_cls, _, _, _ = emd_similarity(None, search_feat, None, gallery_feat[cls_neighbors_index], 0)
        dist = alpha * dist_cls + (1.0 - alpha) * dist_emd
        knn = dist.topk(top_k, largest=True)

    else:
        raise ValueError("Invalid metric specified")

    return knn.indices


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

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.REC, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

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
                reconstruction=False, imgpath=None, keep_rate=None):
        """
        Forward pass of the model.

        Args:
        x (Tensor): Input tensor.
        label (Tensor, optional): Labels of the input samples.
        cam_label (Tensor, optional): Camera labels.
        view_label (Tensor, optional): View labels.
        get_memory (bool, optional): Flag to get memory features.
        memory_global_cls (Tensor, optional): Global class memory.
        memory_local_cls (Tensor, optional): Local class memory.
        gallery_memory_patch (Tensor, optional): Memory patch of gallery samples.
        reconstruction (bool, optional): Flag to enable reconstruction.
        imgpath (str, optional): Path of the image.
        keep_rate (float, optional): Keep rate for dropout.

        Returns:
        Tensor: Output features or logits of the model.
        """

        # Extract base features and indices
        features, idxs = self.base(x, cam_label=cam_label, view_label=view_label, base_keep_rate=keep_rate)

        b1_feat = self.b1(features)  # Shape: [64, N, 768]
        global_feat = b1_feat[:, 0]
        token = features[:, 0:1]
        # Bottleneck features
        feat = self.bottleneck(global_feat)

        # Get gallery memory
        if get_memory:
            global_cls = torch.squeeze(token)
            return features[:, 0:1], features[:, 1:]

        # Reconstruction of occluded patch features
        if reconstruction:
            rec_patch, rec_cls = patch_reconstruction(features[:, 0], features[:, 1:], memory_global_cls, gallery_memory_patch)

            # Reconstruct features
            rec_feat = self.rec_b(torch.cat((rec_cls, rec_patch), dim=1))
            rec_feat = rec_feat[:, 0]
            rec_feat_bn = self.rec_bottleneck(rec_feat)

        # Training branch
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                
                if reconstruction:
                    cls_score_rec = self.rec_classifier(rec_feat_bn)
                    return cls_score_rec, rec_feat, None, None
                
                return cls_score, global_feat, None, None
            
        # Inference branch
        else:
            if self.neck_feat == 'after':
                if reconstruction:
                    return rec_feat_bn, idxs  # Reconstructed feature
                else:
                    return feat, idxs
            else:
                if reconstruction:
                    return rec_feat_bn, idxs  # Reconstructed feature
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
        if cfg.MODEL.REC:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with REC module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
