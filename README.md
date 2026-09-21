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
throughout to prevent data leakage from batch effects.

## Status

🚧 Work in progress. Analysis not completed yet.

## Method

### Preprocessing

Each three-channel image is 
1. resized while preserving its aspect ratio,
2. centre-cropped to 224 x 224 pixels,
3. normalised independently by channel suing the 1st and 99th percentiles.

Training augmentation consists of random horizontal flips, vertical flips, and rotations 
in multiples of 90 degrees. Validation and test images are not augmented.

### Model

The classifier uses an ImageNet-pretrained ResNet-18 backbone. Its final layer is replaced
with a 12-class classification head.

Training uses:
- class-weighted cross-entropy loss,
- AdamW optimisation,
- ReduceLR0nPlateau learning-rate scheduling,
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
│   ├── run_loco.py         #LOCO experiment runner
├── data/                   #Raw BBC021 images and metadata
├── mappings/               #MoA label <--> integer mappings (generated)
│   └── moa_label_map.json
├── results/                #Evaluation outputs (planned)
├── models/                 #Saved model checkpoints (planned)
├── notebooks/
│   ├── 01_eda.ipynb        #Dataset exploration
│   └── 02_results.ipynb    #Results analysis (planned)
├── src/
│   ├── dataset.py          #Loads images, merges image/MoA metadata, drop DMSO
│   ├── preprocessing.py    #Resize/crop, per-channel normalisation, augmentation
│   ├── splitting.py        #LOCO CV folds + compound aware MoA-stratified split
│   ├── model.py            #ResNet-18 backbone with custom MoA classification head
│   ├── train.py            #Training and checkpoint selection
│   ├── evaluate.py         #Prediction, metrics and confidence outputs
│   └── utils.py            #Utility functions
├── requirements.txt
└── README.md
```

