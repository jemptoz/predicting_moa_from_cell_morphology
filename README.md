# Predicting Mechanism of Action from Cell Morphology

## Motivation 

Understanding a drug compound's mechanism of action (MoA) is a key step in
drug discovery, but MoA is often unknown or expensive to determine
experimentally. This project explores whether **mechanism of action can be
predicted directly from fluorescence microscopy images of treated cells**,
using the [BBBC021](https://bbbc.broadinstitute.org/BBBC021) dataset (DAPI,
Tubulin, and Actin channels). Rather than relying on hand-crafted
morphological features, the goal is to train a CNN that learns MoA-relevant
patterns directly from raw images — and to evaluate it using
leave-one-compound-out cross-validation, so it's tested on its ability to
generalise to **unseen compounds**, not just unseen images of compounds it
has already seen. Compound-level and MoA-stratified splitting are used
throughout.


## Results

Three full leave-one-compound-out (LOCO) runs used seeds 42, 43, and 44. Each run trained a separate model for every
held-out compound, producing test prediction for 2,528 images from 38 compounds and 12 MoA classes. For compound
predictions, the 12-class probability vectors were averaged across all images of a compound before selecting the class
with the highest probability.

 | Seed | Image accuracy | Image macro F1 | Compound accuracy | Compound macro F1 |
|:-----|---------------:|---------------:|------------------:|------------------:|
| 42   |          67.1% |          0.719 |     35/38 (92.1%) |             0.925 | 
| 43   |          72.2% |          0.712 |     35/38 (92.1%) |             0.937 | 
| 44   |          74.5% |          0.728 |     36/38 (94.7%) |             0.955 | 

For comparison, a baseline that predicts the most frequent training label in each fold achieved 2.8% image accuracy 
and 2/38 (5.3%) compound accuracy in each run.

The results support prediction of known MoA classes for many previously unseen compounds within this dataset. See
the results notebook for per-class results, confusion matrices, and compound errors.


## Method

### Preprocessing

Each three-channel image is 
1. resized while preserving its aspect ratio,
2. centre-cropped to 224 x 224 pixels,
3. normalised independently by channel using the 1st and 99th percentiles.

Training augmentation consists of random horizontal flips, vertical flips, and rotations 
in multiples of 90 degrees. Validation and test images are not augmented.

### Model

The classifier uses an ImageNet-pretrained ResNet-18 backbone. Its final layer is replaced
with a 12-class classification head.

Training uses:
- class-weighted cross-entropy loss,
- AdamW optimisation,
- ReduceLROnPlateau learning-rate scheduling,
- early stopping based on validation macro F1.

### Leave-one-compound-out (LOCO) evaluation

A separate model is trained for every compound.
1. all images belonging to one compound are held out as the test set,
2. the remaining compounds form the development set,
3. development compounds are divided into training and validation sets,
4. the model is trained from its pretrained initialisation.


## Repository Structure

```
predicting_moa_from_cell_morphology/
├── experiments/
│   ├── run_loco.py             #LOCO experiment runner
├── data/                       
│   ├── resnet18-f37072fd.pth   #ResNet-18 pre-trained weights
│   └── raw/                    #Raw BBC021 images and metadata
├── mappings/                   #MoA label <--> integer mappings (generated)
│   └── moa_label_map.json
├── models/                     #Saved model checkpoints
├── results/                    #Evaluation outputs
├── analysis/                   #Result analysis outputs
├── notebooks/
│   ├── 01_eda.ipynb            #Dataset exploration
│   └── 02_results.ipynb        #Results analysis
├── src/
│   ├── dataset.py              #Loads images, merges image/MoA metadata, drop DMSO
│   ├── preprocessing.py        #Resize/crop, per-channel normalisation, augmentation
│   ├── splitting.py            #LOCO CV folds + compound aware MoA-stratified split
│   ├── model.py                #ResNet-18 backbone with custom MoA classification head
│   ├── train.py                #Training and checkpoint selection
│   ├── evaluate.py             #Prediction, metrics and confidence outputs
│   └── utils.py                #Utility functions
├── requirements.txt
└── README.md
```

## Data setup

Download BBBC021_v1_image.csv, BBBC021_v1_moa.csv, and the image ZIP archives from the [BBBC021 dataset page](https://bbbc.broadinstitute.org/BBBC021).
Place the CSV files in data/raw/ and extract the archives into data/raw/images/, preserving the Week*/Week*_* folders.
The expected layout is:
```
├── data/                       
│   ├── resnet18-f37072fd.pth 
│   └── raw/                    
│       ├── BBBC021_v1_image.csv
│       ├── BBBC021_v1_moa.csv
│       └── images/
│           └── Week1/
│               └── Week1_22123/
│                   └── *.tif
```
The pretrained ResNet-18 weights are included in the repository at data/resnet18-f37072fd.pth.


## Potential extensions

- Compare the CNN with a stronger baseline.
- Evaluate on independently collected images or hold out entire plates to investigate sensitivity to imaging conditions.
- Investigate whether self-supervised pretraining on rest of BBBC021 data would improve final prediction compared to current model pretrained on ImageNet
