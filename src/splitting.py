import numpy as np

def create_loco_folds(dataset, num_folds=None, seed=42):
    """
    Create leave-one-compound-out cross validation folds.

    Each fold holds out all images belonging to one compound as the
    test set and uses all remaining compounds as the development set.

    When "num_folds" is specified, a reproducible subset of compounds
    will be selected for testing. Compounds are selected across MoAs
    so that the test compounds provide broad MoA coverage.

    Parameters
    ----------
    dataset : BBBC021Dataset
        Dataset containing the labelled metadata.
    num_folds : int or None, default=None
        Number of folds to create. If None, use all.
    seed : int, default=42
        Random seed.

    Yields
    -------
    development_indices : list[int]
        Indices of images belonging to the development set.
    test_indices : list[int]
        Indices of images belonging to the test set.
    test_compound : str
        Compound held out for testing.
    """
    all_compounds = dataset.get_compounds()

    if num_folds is not None and not 1 <= num_folds <= len(all_compounds):
        raise ValueError(
            f"num_folds must be between 1 and {len(all_compounds)}, or None"
        )

    if num_folds is None:
        compounds = all_compounds
    else:
        moa_compounds = (
            dataset.metadata
            .groupby("moa")["Image_Metadata_Compound"]
            .unique()
        )

        rng = np.random.default_rng(seed)

        moa_compound_lists = {
            moa: rng.permutation(compound_list).tolist()
            for moa, compound_list in moa_compounds.items()
        }

        selected_compounds = []

        moa_list = list(moa_compounds.keys())
        rng.shuffle(moa_list)
        for moa in moa_list:
            if len(selected_compounds) >= num_folds:
                break
            if moa_compound_lists[moa]:
                selected_compounds.append(
                    moa_compound_lists[moa].pop()
                )
        while len(selected_compounds) < num_folds:
            available_moa = [
                moa
                for moa, compounds_for_moa in moa_compound_lists.items()
                if compounds_for_moa
            ]
            if not available_moa:
                break

            rng.shuffle(available_moa)

            for moa in available_moa:
                if len(selected_compounds) >= num_folds:
                    break

                selected_compounds.append(
                    moa_compound_lists[moa].pop()
                )

        compounds = selected_compounds

    for test_compound in compounds:
        test_indices = dataset.get_indices_for_compound(test_compound)

        development_indices = [
            idx for compound in all_compounds
            if compound != test_compound
            for idx in dataset.get_indices_for_compound(compound)
        ]

        yield development_indices, test_indices, test_compound

def create_validation_split(dataset, development_indices,
                            val_ratio=0.2, min_compounds_per_moa=2, seed=42):
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

    return train_indices, val_indices


