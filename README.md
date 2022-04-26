# CS153 Flower Mapping

Flower mapping final project for CS 153 at Harvey Mudd College.

## Setup

To run this project, first install [pipenv](https://pypi.org/project/pipenv/), then from the root of this repository run `pipenv install`. This will install the required dependencies. Then run `pipenv run [command]` to run a command in the virtual environment (e.g. `pipenv run python src/processing.py`), or `pipenv shell` to activate the environment. The environment should also show up as a Python interpreter option in VSCode.

To download the model initial model weights from https://github.com/kazuto1011/deeplab-pytorch, run `./download-weights.sh`.

## Links

 - Project proposal: [proposal/Hadley_ProjectProposal.pdf](proposal/Hadley_ProjectProposal.pdf)
 - Segmentation polygon script: https://github.com/tommyfuu/flower_map_new/blob/master/scripts/visualizePolygons.py
 - COCO-Stuff: https://github.com/nightrome/cocostuff
 - Pre-trained DeepLab model: https://github.com/kazuto1011/deeplab-pytorch
 - Fine-tuning tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
