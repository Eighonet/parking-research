# Detection of parking space availability based on video
This is a repo that contains the scripts for *''Detection of parking space availability based on video''* thesis and bachelor work.

This repo originated as a fork of a repository from [*''Revising deep learning methods in parking lot occupancy detection''*](https://arxiv.org/abs/2306.04288). I greatly thank the authors for their work. And I recommend checking the repository along with their paperk, their reasearch helped me significantly.

The authors are:
Anastasia Martynova, [Mikhail Kuznetsov](https://github.com/mmkuznecov), [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov), Vladislav Tishin, [Natalia Semenova](https://www.researchgate.net/profile/Natalia-Semenova-7).

And their repo is: [parking-reeach](https://github.com/Eighonet/parking-research)

The original goal of this fork is to port the tools to newest python libraries and to serve as a personal playground  for the work. But after a lot of changes and modifications to the code. It is now used as a proof of concept to the final work.

The training and testing scripts support only a object recognition models for now.

## What hase been added / changed
- I written a training script that lets the user choose how to train a model with the dataset format from their work as both as a training script that uses the testing images from the dataset to test the trained model.
- The creation of your own dataset was a bit simplified and reworked. Consult the readme located in [annotating](annotating/) directory.
- All of the scripts should run both on CPU and GPU.
- Tested on Python 3.11

# Prerequisites

To run the code, you need to install the requirements using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can create and activate the conda enviroment:

```bash
conda env create -f environment.yml
conda activate parking_research
```

The trainig script is setup to log information to [Comet](comet.com).
The training script looks for a api.key file containing a key that you can obtain in your user settings after registering.

# Training
Create a `dataset` directory containing the datasets in the same location as the training script. Add a `api.key` file containing your come API key. Then run the script and follow to onscreen prompts:
```bash 
python train.py
```
For an average size dataset you can use the default values for learning speed, batch size and number of epochs. Always check the validation batch progress to make sure that the model is not overfitting!

# Running tests
After training a model you can test it using the `test.py` script. Run it and follow the prompts:
```bash 
python test.py
```
The testing function can print out the time of one inference and save all of the images that were tested.

# Contact me

If you have some questions about the code, you are welcome to open an issue or contact me through an email. Please do not contact the original authors with questions regarding this fork.

# License

Established code released as open-source software under the MIT license.

# Citation

To be updated.

```
@misc{martynova2023revising,
      title={Revising deep learning methods in parking lot occupancy detection}, 
      author={Anastasia Martynova and Mikhail Kuznetsov and Vadim Porvatov and Vladislav Tishin and Andrey Kuznetsov and Natalia Semenova and Ksenia Kuznetsova},
      year={2023},
      eprint={2306.04288},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
