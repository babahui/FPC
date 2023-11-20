import torch
import torch.nn.functional as F

# copy from https://github.com/anguyen8/deepface-emd/blob/7ca939d773f362fdc7359dab74c7df5035c25d9c/utils/emd.py
def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def emd_similarity(anchor, anchor_center, fb, fb_center, stage, method=''):
    flows = None
    u = v = None
    if stage == 0:  # stage 1: Cosine similarity
        sim = torch.einsum('c,nc->n', anchor_center, fb_center)
        # print('sim: ', sim)

    else:  # stage 2: re-ranking with EMD
        _, R_m = anchor.size()
        N, _, R_s = fb.size()

        sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R_s, R_m) # R_s, R_m
        # print('sim: ', sim)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)
        # print('K: ', K)

        if method == 'uniform':
            u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
            v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
        elif method == 'sc':
            u = torch.sum(dis, 2)
            u = u / (u.sum(dim=1, keepdims=True) + 1e-7)
            v = torch.sum(dis, 1)
            v = v / (v.sum(dim=1, keepdims=True) + 1e-7)
        elif method == 'apc':
            att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R_s)
            u = att / (att.sum(dim=1, keepdims=True) + 1e-7)

            att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R_m)
            v = att / (att.sum(dim=1, keepdims=True) + 1e-7)
        elif method == 'uew':
            att1 = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
            att2 = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
            s = att1.sum(dim=1, keepdims=True) + att2.sum(dim=1, keepdims=True) + 1e-7
            u = att1 / s
            v = att2 / s
        else:
            print('No found method.')
            exit(0)

        
        T = Sinkhorn(K, u, v)
        # print(T)
        sim = torch.sum(T * sim, dim=(1, 2))
        sim = torch.nan_to_num(sim) 
        flows = T
        
    return sim, flows, u, v