import torch
import torch.nn.functional as F
import math

def index_to_log_onehot(x, num_classes):
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

num_embed = 5
mask_class = num_embed - 1

x_t = torch.tensor([
    [0, 1, 4],
    [2, 4, 3]
])

a = torch.tensor(math.log(0.1))
b = torch.tensor(math.log(0.2))
c = torch.tensor(math.log(0.5))

res = index_to_log_onehot(x_t, num_embed)[:, :-1, :]

res = (res + a).logaddexp(b)

mask_class_mask = x_t == mask_class
mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, num_embed - 1, -1)
res[mask_class_mask] = c



##########
bsz, num_latent_pixels = x_t.shape

res = torch.empty((bsz, num_embed-1, num_latent_pixels))

mask_class_mask = x_t == mask_class
mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, num_embed - 1, -1)

res[mask_class_mask] = c

non_mask_class_mask = ~mask_class_mask

res[non_mask_class_mask] = b

xres[torch.tensor([[0, 0, 0], [0, 1, 1]])] = a + b
res[x_t[non_mask_class_mask], non_mask_class_mask] = a + b
