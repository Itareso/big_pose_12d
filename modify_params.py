import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    "ResNet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "ResNet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "ResNet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "ResNet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "ResNet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

model_res34 = model_zoo.load_url(model_urls["ResNet34"])

print(type(model_res34))

print(model_res34.keys())

conv1_weight = model_res34["conv1.weight"]
conv1_weight1 = conv1_weight.clone()
conv1_weight2 = conv1_weight.clone()
conv1_weight_new = torch.cat((conv1_weight, conv1_weight1, conv1_weight2), dim=1)
print(conv1_weight_new.shape)
model_res34["conv1.weight"] = conv1_weight_new

torch.save(model_res34, "checkpoints/resnet34_pretrained.pth.tar")
