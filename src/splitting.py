import numpy as np

def create_loco_folds(dataset):
    """
    Create leave-one-compound-out cross validation folds.

    Each fold holds out all images belonging to one compound as the
    test set and uses all remaining compounds as the development set.

    Parameters
    ----------
    dataset : BBBC021Dataset
        Dataset containing the labelled metadata.

    Yields
    -------
    development_indices : list[int]
        Indices of images belonging to the development set.
    test_indices : list[int]
        Indices of images belonging to the test set.
    test_compound : str
        Compound held out for testing.
    """

    compounds = dataset.get_compounds()

    for test_compound in compounds:
        test_indices = dataset.get_indices_for_compound(test_compound)
        development_indices = [
            idx for compound in compounds
            if compound != test_compound
            for idx in dataset.get_indices_for_compound(compound)
        ]

        yield development_indices, test_indices, test_compound

def create_validation_split(dataset, development_indices,
                            val_ratio=0.2, min_compounds_per_moa=2, seed=42,
                            verbose=False):
    """
    Create validation and training splits from the development set using
    compound-based stratification.
    This function ensures that entire compounds are held out for validation,
    and not just individual images, to prevent data leakage. It also maintains
    stratification by MoA so that each MoA is represented in both training and
    validation sets when possible.

    Function follows these rules:
    1. For each MoA, calculate how many compounds to hold out for
    validation n_val.
    2. Only hold compounds if enough remain for training.
    3. If an MoA has too few compounds, keep all for training.
    4. Randomly select which compounds go to validation (reproducible with seed).

    Parameters
    ----------
    dataset : BBBC021Dataset
        Full BBBC021 dataset.
    development_indices : list[int]
        Indices belonging to the development set.
    val_ratio : float, default=0.2
        Proportion of compounds per MoA to hold out for validation.
        Must be between 0 and 1.
    min_compounds_per_moa : int, default=2
        Minimum number of compounds that must remain in the training
        set for each MoA.
    seed : int, default=42
        Random seed for reproducible compound selection.

    Returns
    -------
    train_indices : list[int]
        Indices of images for the training set.
    val_indices : list[int]
        Indices of images for the validation set.

    """

    development_metadata = dataset.metadata.iloc[development_indices].copy()

    moa_compounds = (
        development_metadata
        .groupby("moa")["Image_Metadata_Compound"]
        .unique()
    )

    rng = np.random.default_rng(seed)

    train_compounds = []
    val_compounds = []

    for moa, compounds in moa_compounds.items():
        n_compounds = len(compounds)
        n_val = max(1, int(n_compounds * val_ratio))

        if n_compounds - n_val >= min_compounds_per_moa:
            selected = rng.choice(compounds, n_val, replace=False).tolist()
            val_compounds.extend(selected)
            train_compounds.extend([c for c in compounds if c not in selected])
        else:
            train_compounds.extend(compounds)

    train_indices = development_metadata[
        development_metadata["Image_Metadata_Compound"].isin(train_compounds)
    ].index.tolist()
    val_indices = development_metadata[
        development_metadata["Image_Metadata_Compound"].isin(val_compounds)
    ].index.tolist()

    if verbose:
        print(f"\nSplit Summary:")
        print(f"  Training compounds: {len(set(train_compounds))}")
        print(f"  Validation compounds: {len(set(val_compounds))}")
        print(f"  Training images: {len(train_indices)}")
        print(f"  Validation images: {len(val_indices)}")
        print(f"  Total images: {len(train_indices) + len(val_indices)}")

    return train_indices, val_indices


