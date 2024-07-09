# import torch
# from torchvision.models import efficientnet_v2_s

# model = efficientnet_v2_s(weights='IMAGENET1K_V1')
# print(model)


import timm

model = timm.create_model('convnext_base', pretrained=True)
print(model)
