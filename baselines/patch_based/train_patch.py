from baselines.utils.common_utils import seed_everything, get_device
from baselines.utils.patch_utils import *
from conv_based_models import *
from transformer_based_models import *
from spt_lsa_vit import *
from cfen import *


seed_everything(seed=42)
device = get_device()
args = parse_args()

if args.model_type == 'alexnet':
    model, loss_func, optimizer = get_alexnet()
elif args.model_type == 'carnet':
    model, loss_func, optimizer = get_carnet()
elif args.model_type == 'efficientnet':
    model, loss_func, optimizer = get_efficientnet()
elif args.model_type == 'malexnet':
    model, loss_func, optimizer = get_malexnet()
elif args.model_type == 'mobilenet':
    model, loss_func, optimizer = get_mobilenet()
elif args.model_type == 'resnet':
    model, loss_func, optimizer = get_resnet50()
elif args.model_type == 'vgg16':
    model, loss_func, optimizer = get_vgg16()
elif args.model_type == 'vgg19':
    model, loss_func, optimizer = get_vgg19()
elif args.model_type == 'deit':
    model, loss_func, optimizer = get_deit()
elif args.model_type == 'pit':
    model, loss_func, optimizer = get_pit()
elif args.model_type == 'pretrained_vit':
    model, loss_func, optimizer = get_pretrained_vit()
elif args.model_type == 'regular_vit':
    model, loss_func, optimizer = get_regular_vit()
elif args.model_type == 'spt_lsa_vit':
    model, loss_func, optimizer = get_spt_lsa_vit()


if args.model_type != 'cfen':
    train_data, valid_data, test_data = load_data(args.path, 32, image_transforms=image_transforms)
    model.to(device)
    train_patch_model(model, loss_func, optimizer, train_data, valid_data)
else:
    train_data, valid_data = load_cfen_data(args.path)
    model, criterion, CEloss, optimizer = get_cfen()
    model.to(device)
    train_cfen_model(model, criterion, CEloss, optimizer, train_data, valid_data)