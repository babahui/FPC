from PIL import Image
import torch
from visualize_mask import get_real_idx, mask, save_img_batch
from evit import deit_small_patch16_shrink_base
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

if __name__=="__main__":
    
    t_resize_crop = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ])

    t_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    model_name='deit_small_patch16_shrink_base'
    base_keep_rate=0.9
    drop_loc='(3, 6, 9)'
    nb_classes= 1000
    model = deit_small_patch16_shrink_base(base_keep_rate=0.7)
    checkpoint = torch.load("checkpoints/evit-0.9-fuse-img224-deit-s.pth", map_location='cuda:0')['model']
    model.load_state_dict(checkpoint)
    image = Image.open("images/dog.jpg")
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(3, 1, 1).cuda()
    images = t_resize_crop(image)
    images = t_to_tensor(images).unsqueeze(0).cuda()
    model.cuda()
    x, idxs= model(images,get_idx=True)
    images = images * std + mean
    B = images.size(0)
    output_dir="images/"
    idxs = get_real_idx(idxs, fuse_token=True)
    ii = 0
    
    for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx= B * ii +  B)

    save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx= B * ii +  B)