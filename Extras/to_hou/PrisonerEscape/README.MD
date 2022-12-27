"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
# Fugitive Project

This repository is for inferring the motion of a smart, sneaking, evading, 
and deceptive prisoner. Within this repository, we have the capability
to simulate fugitive escape scenarios (see the simulator folder) and predict the current location of the fugitive based on past blue observation (filtering) and predict the future location of the prisoner based on past fugitive locations(prediction). 

### Installation
After cloning the repository, please use 

Use the provided conda environment (`environment.yml`) file to install dependencies:
 conda env create -f conda_environment.yml

This will create an environment named 'prisoner'; edit the first line of `conda_environment.yml` to call it something else.

### Simulator
Within our environment, we have several classes to represent the terrain, 
different objects (town, camera, etc.), and step all moving objects based 
on policies/heuristics. If you would like details of our prisoner_env configuration
(state space, observation space, action space, etc.), please refer to [this file](./simulator/prisoner_env.py)

Note within the simulator, there is many options that can be changed. One key 
option is the rendering mechanism. We have a fast option that is less aesthetic,
and a slow option that is more aesthetic.
For training, do not render as this will result in slowdown.
For debugging, debug with the fast option.
For results, use the slow rendering to get best display.

### Collecting the Dataset
The dataset is saved as a numpy array that consists of 5 components: 
blue observations, red observations, red locations, and dones. The dones mark 
the end of each episode. Run `datasets/collect_demonstrations.py` to create two numpy files that will be used as the train and test set for the models. 

### Training the Model
To train various models, run `train.py --config configs/filtering.yaml` and `train.py --config configs/prediction.yaml`. 


### Using Docker
run `docker build . -t torch/geometric`
then to launch container with gpu access, use `docker run -it -P --name fugitive -v /home/sye40/PrisonerEscape:/workspace/PrisonerEscape -v /star-data/sye40:/workspace/star-data --gpus all torch/geometric`

Then to start it use `docker start fugitive`
After starting it you can enter with `docker attach fugitive`
