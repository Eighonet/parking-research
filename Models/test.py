from baselines.utils.inter_utils import *
from baselines.intersection_based.inter_models import *
from baselines.utils.common_utils import seed_everything, get_device
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import inquirer

# parser = argparse.ArgumentParser(description="Test model on a dataset")
# parser.add_argument("-d", "--dataset", type=str, help="Name of a dataset located in the datasets directory")
# parser.add_argument("-m", "--model", type=str, help="Path to a model to test")
# parser.add_argument("-t", "--type", type=str, choices=models ,help="Type of a model to test")
# parser.add_argument("--pretrained", type=bool, help="Was the model pretrained?", default=False)
# parser.add_argument("-s", "--save", type=bool, help="Export prediciton images", default=False)
# parser.add_argument("--timeit", type=bool, help="Time one inference")

questions = [
    
    inquirer.Confirm('pretrained', message = "Using a pretrained model?", default=False),
    inquirer.List('model',
                  message="What model type?",
                  choices = ["faster_rcnn_mobilenet", "faster_rcnn_mobilenetV3_Large", "faster_rcnn_mobilenetV3_Small", "faster_rcnn_resnet", "faster_rcnn_vgg", "retinanet_mobilenet", "retinanet_resnet", "retinanet_vgg", "retinanet_mobilenetV3_Small", "retinanet_mobilenetV3_Large"],
                  ignore = lambda x: x["pretrained"] == True),
    
    inquirer.List('model',
                  message="What model do type?",
                  choices = ["faster_rcnn_mobilenetV3_Large", "faster_rcnn_resnet", "retinanet_resnet"],
                  ignore = lambda x: x["pretrained"] == False),
    
    inquirer.Path('path',
                 message="Path to a model to test",
                 path_type=inquirer.Path.FILE,
                 exists = True
                ),
    
    inquirer.Text('dataset',
                  message="What dataset to use for testing?"),
    inquirer.Confirm('timeit', message = "Show inference time?", default=True),
    inquirer.Confirm('save', message = "Save inference to images?", default=True),
]
answers = inquirer.prompt(questions, raise_keyboard_interrupt=True)

warnings.filterwarnings("ignore")
device = get_device()

dataset = answers["dataset"]

dataset_path = os.path.join("datasets/"+dataset, dataset)
#Get wanted model from inter models 
if answers["model"] == 'faster_rcnn_mobilenet':
    model = get_model(faster_rcnn_mobilenet_params, answers["pretrained"])
elif answers["model"] == 'faster_rcnn_mobilenetV3_Large':
    model = get_model(faster_rcnn_mobilenetV3_Large_params, answers["pretrained"])
elif answers["model"] == 'faster_rcnn_mobilenetV3_Small':
    model = get_model(faster_rcnn_mobilenetV3_Small_params, answers["pretrained"])
elif answers["model"] == 'faster_rcnn_resnet':
    model = get_model(faster_rcnn_resnet_params, answers["pretrained"])
elif answers["model"] == 'faster_rcnn_vgg':
    model = get_model(faster_rcnn_vgg_params, answers["pretrained"])
elif answers["model"] == 'retinanet_mobilenet':
    model = get_model(retinanet_mobilenet_params, answers["pretrained"])
elif answers["model"] == 'retinanet_resnet':
    model = get_model(retinanet_resnet_params, answers["pretrained"])
elif answers["model"] == 'retinanet_vgg':
    model = get_model(retinanet_vgg_params, answers["pretrained"])
elif answers["model"] == 'retinanet_mobilenetV3_Large':
    model = get_model(retinanet_mobilenetV3_Large_params, answers["pretrained"])
elif answers["model"] == 'retinanet_mobilenetV3_Small':
    model = get_model(retinanet_mobilenetV3_Small_params, answers["pretrained"])
else:
    raise Exception('Invalid model type')

load_model(model, device, answers["path"])
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
    batch_size= 1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
#show_from_dataset(2, test_data_loader)
acc_list = test_model(model, device, test_data_loader, 0.85, save=answers["save"], timeit=answers["timeit"])
avg_acc = sum(acc_list) / len(acc_list)
print("Average accuracy on test dataset: %0.2f %%" %(avg_acc*100))