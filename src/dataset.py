from pathlib import Path
import pandas as pd
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset
from src.utils import create_moa_label_mapping


class BBC021Dataset(Dataset):


    def __init__(self,
                 image_csv,
                 moa_csv,
                 image_root,
                 label_mapping=None,
                 transform=None):
        """
        Initialise the BBBC021 dataset.

        Parameters
        -------
        image_csv : str or Path
            Path to BBBC021_v1_image.csv.
        moa_csv : str or Path
            Path to BBBC021_v1_moa.csv.
        image_root : str or Path
            Root directory containing the microscopy images.
        label_mapping : dict or None
                Mapping from MoA class names to integer class indices used by PyTorch.
        transform : callable, optional
            Transformations applied to each image.
        """
        self.image_root = Path(image_root)
        self.transform = transform

        image_metadata = pd.read_csv(image_csv)
        moa_metadata = pd.read_csv(moa_csv)

        self.metadata = image_metadata.merge(moa_metadata,
                         left_on=[
                             "Image_Metadata_Compound",
                             "Image_Metadata_Concentration"
                         ],
                         right_on=[
                             "compound",
                             "concentration"
                         ],
                          how="inner")

        self.metadata = self.metadata[
            self.metadata["moa"] != "DMSO"
        ].reset_index(drop=True)

        if label_mapping is None:
            self.label_mapping = create_moa_label_mapping(self.metadata)
        else:
            self.label_mapping = label_mapping

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Return one image and its MoA label."""
        row = self.metadata.iloc[idx]

        dapi_path = (
            self.image_root
            / row["Image_PathName_DAPI"]
            / row["Image_FileName_DAPI"]
        )

        tubulin_path = (
                self.image_root
                / row["Image_PathName_Tubulin"]
                / row["Image_FileName_Tubulin"]
        )

        actin_path = (
            self.image_root
            / row["Image_PathName_Actin"]
            / row["Image_FileName_Actin"]
        )

        dapi = tifffile.imread(dapi_path)
        tubulin = tifffile.imread(tubulin_path)
        actin = tifffile.imread(actin_path)

        image = torch.from_numpy(np.stack([dapi, tubulin, actin])).float()

        label = self.label_mapping[row["moa"]]

        return image, label

    def get_indices_for_compound(self, compound):
        """Return dataset index corresponding to given compound."""

        indices = self.metadata.index[
            self.metadata["Image_Metadata_Compound"] == compound
            ].tolist()

        if not indices:
            raise ValueError(f"Compound {compound} not found in dataset.")

        return indices

    def get_compounds(self):
        """Return the unique compounds in the dataset."""
        return self.metadata["Image_Metadata_Compound"].unique()
