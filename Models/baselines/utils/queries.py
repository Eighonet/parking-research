#This file contains the settings for the inquirer package for questioning the user for user defined parameters

import inquirer
from inquirer import errors

#Cheks if a string is a float number
def check_float(answers, current):
    try:
        float(current)
    except  ValueError:
        raise errors.ValidationError('', reason='must be a float!')
    return True

#Checks if a string is an integer
def check_int(answers, current):
    if not current.isdigit():
        raise errors.ValidationError('', reason='must be an int!')
    else:
        return True 

questions = [
    
    inquirer.Confirm('pretrained', message = "Use pretrained model?", default=False),
    inquirer.List('model',
                  message="What model do you want to train?",
                  choices = ["faster_rcnn_mobilenet", "faster_rcnn_mobilenetV3_Large", "faster_rcnn_mobilenetV3_Small", "faster_rcnn_resnet", "faster_rcnn_vgg", "retinanet_mobilenet", "retinanet_resnet", "retinanet_vgg", "retinanet_mobilenetV3_Small", "retinanet_mobilenetV3_Large"],
                  ignore = lambda x: x["pretrained"] == True),
    
    inquirer.List('model',
                  message="What model do you want to train?",
                  choices = ["faster_rcnn_mobilenetV3_Large", "faster_rcnn_resnet", "retinanet_resnet"],
                  ignore = lambda x: x["pretrained"] == False),
    
    inquirer.Confirm('retrain', message = "Retrain an existing model?", default=False),
    inquirer.Path('saved',
                 message="Path to a saved model to retrain",
                 path_type=inquirer.Path.FILE,
                 exists = True,
                 ignore = lambda x: x["retrain"] == False
                ),
    
    inquirer.Text('dataset_n',
                  message="How many datasets do you want to use?", validate = check_int, default = 1),
    
    inquirer.Text('batch',
                  message="Size of a single batch",
                  default = 2,
                  validate = check_int),
    
    inquirer.Text('rate',
                  message="Learning rate",
                  default = 0.001,
                  validate = check_float),
    
    inquirer.Confirm('warmup', message = "Use a warming up scheduler for first epoch? (recomended for new datasets)", default=False),
    
    inquirer.Text('save_rate',
                message="How often to save localy? (num of epoch)",
                default = 20,
                validate = check_int),
    
    inquirer.Text('name',
                  message="How to name the experiment in comet?")
]