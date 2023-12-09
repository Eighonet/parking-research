import torch
import torch.nn as nn
import torch.optim as optim
import pit
from transformers import ViTFeatureExtractor, ViTForImageClassification
from vit_pytorch import ViT

num_classes = 2

def get_deit():
    
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(in_features=768, out_features=2, bias=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    return model, loss_func, optimizer

def get_pit():

    model = pit.pit_s(pretrained=False)
    model.load_state_dict(torch.load('./weights/pit_s_809.pth'))

    # get weights from https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/pit.py

    model.head = nn.Linear(in_features=576, out_features=2, bias=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    return model, loss_func, optimizer

def get_pretrained_vit():

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')
    pretrained_vit = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')

    for param in pretrained_vit.parameters():
        param.requires_grad = False

    pretrained_vit.classifier = nn.Sequential(
        nn.Linear(1024, num_classes))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_vit.parameters(), lr=0.0003)

    return pretrained_vit, loss_func, optimizer

def get_regular_vit():

    vit = ViT(
        image_size = 384,
        patch_size = 32,
        num_classes = 2,
        dim = 64,
        depth = 16,
        heads = 36,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit.parameters())

    return vit, loss_func, optimizer