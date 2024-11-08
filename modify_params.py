import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# model_urls = {
#     "ResNet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
#     "ResNet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
#     "ResNet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
#     "ResNet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
#     "ResNet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
# }

# model_res34 = model_zoo.load_url(model_urls["ResNet34"])

# print(type(model_res34))

# print(model_res34.keys())

# conv1_weight = model_res34["conv1.weight"]
# conv1_weight1 = conv1_weight.clone()
# conv1_weight2 = conv1_weight.clone()
# conv1_weight3 = conv1_weight.clone()
# conv1_weight4 = conv1_weight.clone()
# conv1_weight_new = torch.cat((conv1_weight, conv1_weight1, conv1_weight2, conv1_weight3, conv1_weight4), dim=1)
# print(conv1_weight_new.shape)
# model_res34["conv1.weight"] = conv1_weight_new

# print(model_res34["conv1.weight"].shape)
# torch.save(model_res34, "checkpoints/resnet34_pretrained.pth.tar")

file = "checkpoints/HybridBaseline1011.pth.tar"
model = torch.load(file)
print(model.keys())
box1 = model['box_head_kin.layers.0.weight']
box2 = box1.clone()
box3 = box1.clone()
box4 = box1.clone()
box5 = box1.clone()

# box_3 = torch.cat((box1, box2, box3), dim = 1)
# box_5 = torch.cat((box1, box2, box3, box4, box5), dim = 1)
# model['box_head_kin.layers.0.weight'] = box_3
# torch.save(model, "checkpoints/HybridBaseline1011_3.pth.tar")
# model['box_head_kin.layers.0.weight'] = box_5
# torch.save(model, "checkpoints/HybridBaseline1011_5.pth.tar")


