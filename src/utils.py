# Loss function
import os 
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/Users/kaushalk/Desktop/open_projects/Telengana_Crop_Health/')
# Add the 'src' directory to the Python path
from src.process_data import compute_indices_for_df, prepare_model_data


# EVI calculation
def calculate_evi(nir, red, blue):
    """
    Calculate the Enhanced Vegetation Index (EVI) using the formula:
    EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
    """
    # Define coefficients for EVI calculation
    G = 2.5    # Gain factor
    C1 = 6.0   # Coefficient for red band
    C2 = 7.5   # Coefficient for blue band
    L = 1.0    # Canopy background adjustment
    EPSILON = 1e-10  # Small constant to avoid division by zero

    # Ensure input bands are floats to prevent integer division
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)

    # Compute the EVI using the formula
    numerator = nir - red
    denominator = nir + C1 * red - C2 * blue + L + EPSILON
    evi = G * (numerator / denominator)

    # Clip the EVI values to the valid range [-1, 1]
    evi = np.clip(evi, -1, 1)

    return evi

#Compute all indices values for dataframe

def compute_indices(tif_path):
    """
    Reads a 10-band Sentinel-2 GeoTIFF from `tif_path` and computes vegetation indices.
    Returns a dictionary of mean index values.
    """
    # Open the 10-band DataArray (assuming 'band' is the first dimension)
    data_array = xr.open_dataarray(tif_path)

    # Extract each band by index (assuming band=0 -> B2, band=1 -> B3, etc.)
    B2  = data_array.isel(band=0).astype(float)  # Blue
    B3  = data_array.isel(band=1).astype(float)  # Green
    B4  = data_array.isel(band=2).astype(float)  # Red
    B5  = data_array.isel(band=3).astype(float)  # Red Edge
    B6  = data_array.isel(band=4)
    B7  = data_array.isel(band=5)
    B8  = data_array.isel(band=6).astype(float)  # NIR (wide)
    B8A = data_array.isel(band=7).astype(float)  # NIR (narrow)
    B11 = data_array.isel(band=8).astype(float)  # SWIR1
    B12 = data_array.isel(band=9).astype(float)  # SWIR2

    # Normalize bands by dividing by 10000 (to match Sentinel-2 reflectance scaling)
    B2 /= 10000
    B3 /= 10000
    B4 /= 10000
    B5 /= 10000
    B8 /= 10000
    B8A /= 10000
    B11 /= 10000

    # --- Compute Indices (all band computations in float) ---
    EPSILON = 1e-10  # To avoid division by zero

    # 2. EVI (with normalized bands): 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    #evi = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0 + EPSILON)
    evi = calculate_evi(B8, B4, B2)

    # 4. GNDVI: (NIR - GREEN) / (NIR + GREEN)
    gndvi = (B8 - B3) / (B8 + B3 + EPSILON)

    # 7. Moisture Index: (B8A - B11) / (B8A + B11)
    moisture_idx = (B8A - B11) / (B8A + B11 + EPSILON)
    
    # 8. NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    ndre = (B8 - B5) / (B8 + B5 + EPSILON)

    # 9. CCCI: NDRE / NDVI
    lswi = (B8 - B11) / (B8 + B11 + EPSILON)  # Add EPSILON to avoid zero division

    # Leaf chloryphil 
    lci = (B8 - B5) / (B8 + B4 + EPSILON)
    
    # --- Compute means for each index across the array ---
    result = {
        'EVI': np.nanmean(evi),
        'GNDVI': gndvi.mean().item(),
        'MoistureIndex': moisture_idx.mean().item(),
        'NDRE': ndre.mean().item(),
        'LSWI': lswi.mean().item(),
        'LCI': lci.mean().item()
    }

    # Close the data to free memory
    data_array.close()
    
    return result


def prepare_indeces_data_stats(qtr_file: str, indices_file: str, target_substrings=None, columns_to_drop=None):
    if target_substrings is None:
        target_substrings = [f"M{i}" for i in range(1, 24)]
    if columns_to_drop is None:
        columns_to_drop = ['State', 'District', 'Sub-District', 
                           'SDate', 'HDate', 'geometry', 'tif_path']

    # Load data
    qdf = pd.read_csv(qtr_file, index_col=0)
    idf = pd.read_csv(indices_file, index_col=0)

    # Merge data on 'FarmID'
    df = pd.merge(idf, qdf, on="FarmID", how="inner", suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.endswith('_drop')]
    
    # Preprocess date columns and compute time difference
    df['SDate'] = pd.to_datetime(df['SDate'])
    df['HDate'] = pd.to_datetime(df['HDate'])
    df['TDays'] = (df['HDate'] - df['SDate']).dt.days
    
    # Drop unnecessary columns
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Compute statistical features
    for suffix in {col.split('_', 1)[1] for col in df.columns if any(sub in col for sub in target_substrings)}:
        matching_columns = [col for col in df.columns if suffix in col and any(sub in col for sub in target_substrings)]
        
        df[f"M_{suffix}_mean"] = df[matching_columns].mean(axis=1, skipna=True)
        df[f"M_{suffix}_std"] = df[matching_columns].std(axis=1, skipna=True)
        df[f"M_{suffix}_skew"] = df[matching_columns].apply(lambda x: skew(x, nan_policy='omit'), axis=1)
        df[f"M_{suffix}_kurtosis"] = df[matching_columns].apply(lambda x: kurtosis(x, nan_policy='omit'), axis=1)
        df[f"M_{suffix}_diff1_mean"] = df[matching_columns].diff(axis=1).mean(axis=1, skipna=True)
        df[f"M_{suffix}_diff2_mean"] = df[matching_columns].diff(axis=1).diff(axis=1).mean(axis=1, skipna=True)

    # Drop original M1-M23 columns after feature extraction
    df = df.drop(columns=[col for col in df.columns if any(sub in col for sub in target_substrings)])

    # Prepare final dataset
    train_df = prepare_model_data(df)

    return train_df



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for handling class imbalance.
        :param alpha: Class weights (tensor), helps address class imbalance
        :param gamma: Focusing parameter to down-weight easy examples
        :param reduction: 'mean' or 'sum' for loss aggregation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none') if alpha is not None else nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)  # Compute CE loss
        pt = torch.exp(-ce_loss)  # Compute pt = exp(-CE)
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Compute Focal Loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss  # If no reduction, return as-is