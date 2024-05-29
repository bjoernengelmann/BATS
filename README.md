# BATS: BenchmArking Text Simplification

#### Setup
For setting up the Docker container please run the following commands:

> docker build -t bats .

> docker-compose up

#### ARTS Datasets

ARTS94, ARTS300 and ARTS3000 can be found on [Zenodo](https://zenodo.org/records/11371690).

#### Data Structure

In the folder *vectors* the pickled initial 1249-dimensional vectors of all datasets used for our evaluation can be found. Additionally, the folder contains pickled required data for ARTS.
In the folder *additionalData* the additional external resources, e.g., those used in by the labeling functions for constructing our vectors can be found.

#### Labeling Functions
All implementations of our labeling functions are in workspace/labeling_functions.py.

#### Experiments
To re-run our experiments please refer to the three notebooks RQ1, RQ2, and RQ3.
