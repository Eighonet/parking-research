from baselines.utils.inter_utils import *
from baselines.intersection_based.inter_models import *
from baselines.utils.common_utils import seed_everything, get_device
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import argparse

parser = argparse.ArgumentParser(description="Test model on a dataset")
parser.add_argument("-d", "--dataset", type=str, help="Name of a dataset located in the datasets directory")
parser.add_argument("-m", "--model", type=str, help="Path to a model to test")
parser.add_argument("-t", "--type", type=str, choices=models ,help="Type of a model to test")
parser.add_argument("--pretrained", type=bool, help="Was the model pretrained?", default=False)
parser.add_argument("-s", "--save", type=bool, help="Export prediciton images", default=False)
args = parser.parse_args()

warnings.filterwarnings("ignore")
device = get_device()

dataset = args.dataset

dataset_path = os.path.join("datasets/"+dataset, dataset)
#Get wanted model from inter models 
if args.type == 'faster_rcnn_mobilenet':
    model = get_model(faster_rcnn_mobilenet_params, args.pretrained)
elif args.type == 'faster_rcnn_mobilenetV3_Large':
    model = get_model(faster_rcnn_mobilenetV3_Large_params, args.pretrained)
elif args.type == 'faster_rcnn_mobilenetV3_Small':
    model = get_model(faster_rcnn_mobilenetV3_Small_params, args.pretrained)
elif args.type == 'faster_rcnn_resnet':
    model = get_model(faster_rcnn_resnet_params, args.pretrained)
elif args.type == 'faster_rcnn_vgg':
    model = get_model(faster_rcnn_vgg_params, args.pretrained)
elif args.type == 'retinanet_mobilenet':
    model = get_model(retinanet_mobilenet_params, args.pretrained)
elif args.type == 'retinanet_resnet':
    model = get_model(retinanet_resnet_params, args.pretrained)
elif args.type == 'retinanet_vgg':
    model = get_model(retinanet_vgg_params, args.pretrained)
load_model(model, device, args.model)
model.to(device)

DIR_INPUT = os.path.join(dataset_path, 'splitted_images')
DIR_TEST = f'{DIR_INPUT}/test'
dataframe = pd.read_csv("datasets/"+str(dataset)+"/"+str(dataset)+"/"+str(dataset)+"_dataframe.csv")

test_df = get_testDataframe(dataframe)

# dataloaders
test_dataset = ParkDataset(test_df, DIR_TEST, get_valid_transform())

# Make a testing DataLoader
test_data_loader = DataLoader(
    test_dataset,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
#show_from_dataset(2, test_data_loader)
acc_list = test_model(model, device, test_data_loader, 0.85, save=args.save)
avg_acc = sum(acc_list) / len(acc_list)
print("Average accuracy on test dataset: %0.2f %%" %(avg_acc*100))