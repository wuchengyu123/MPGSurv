## A Novel Multi-modal Population-graph based Framework for Patients of Esophageal Squamous Cell Cancer Prognostic Risk Prediction

## A Quick Overview 

<img width="600" height="450" src="./framework.png">


## Setup
### Requirements
* Linux (tested on Ubuntu 16.04, 18.04, 20.04)
* Python 3.6+
* PyTorch 1.6 or higher (tested on PyTorch 1.13.1)
* CUDA 11.3 or higher (tested on CUDA 11.6+torch-geometric 2.2.0)

### Installation
  
``conda env create -f environment.yml``

## Training and Evaluation

The training and evaluation code of stage 1 and 2 can be overviewed in  ``main_stage1.py`` and ``main_stage2.py``. The code of proposed MPGSurv model can be seen in  ``/models``.

## Dataset

we apologize that we do not have the right to disclose the datasets collected by the hospital. Due to ethical review and patient privacy disclosure restrictions, we are unable to request public disclosure of the dataset at this time. In addition, future work will be based on this dataset and it is not appropriate to disclose the dataset at this time. The data types and requirements can be set according to ``/dataloader``.


## Acknowlegment

Our repo is developed based on the these projects: [MMMNA-Net](https://github.com/TangWen920812/mmmna-net), [GraphSAGE](https://github.com/twjiang/graphSAGE-pytorch)
