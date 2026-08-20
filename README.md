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

🚧 Work in progress. Data pipeline, preprocessing, and splitting logic are
implemented. Model training and evaluation are not yet complete.

## Repository Structure

```
predicting_moa_from_cell_morphology/
├── data/   #Raw BBC021 images and metadata
├── mappings/   #MoA label <--> integer mappings (generated)
├── results/    #Evaluation outputs (planned)
├── models/     #Saved model checkpoints (planned)
├── notebooks/
│   ├── 01_eda.ipynb        #Exploratory data analysis
│   └── 02_results.ipynb    #Results and analysis 
├── src/
│   ├── dataset.py          #Loads images, merges image/MoA metadata, drop DMSO
│   ├── splitting.py        #LOCO CV folds + compound aware MoA-stratified split
│   ├── preprocessing.py    #Resize/crop, per-channel normalisation, augmentation
│   ├── model.py            #ResNet-18 backbone with custom MoA classification head
│   ├── train.py            #Training loop (planned)
│   ├── evaluate.py         #Model evaluation (planned)
│   └── utils.py            #Utility functions
├── README.md
└── requirements.txt
```

