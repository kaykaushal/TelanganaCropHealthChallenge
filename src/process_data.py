import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import xarray as xr

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
    
    # 1. NDVI: (NIR - RED) / (NIR + RED)
    ndvi = (B8 - B4) / (B8 + B4 + EPSILON)

    # 2. EVI (with normalized bands): 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    evi = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0 + EPSILON)

    # 3. NDWI: (GREEN - NIR) / (GREEN + NIR)
    ndwi = (B3 - B8) / (B3 + B8 + EPSILON)

    # 4. GNDVI: (NIR - GREEN) / (NIR + GREEN)
    gndvi = (B8 - B3) / (B8 + B3 + EPSILON)

    # 5. SAVI: (1 + L) * (NIR - RED) / (NIR + RED + L)
    L = 0.5  # Soil adjustment factor
    savi = (1.0 + L) * (B8 - B4) / (B8 + B4 + L + EPSILON)

    # 6. MSAVI: (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - RED))) / 2
    msavi = (2.0 * B8 + 1.0 - np.sqrt((2.0 * B8 + 1.0)**2 - 8.0 * (B8 - B4))) / 2.0

    # 7. Moisture Index: (B8A - B11) / (B8A + B11)
    moisture_idx = (B8A - B11) / (B8A + B11 + EPSILON)
    
    # 8. NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    ndre = (B8 - B5) / (B8 + B5 + EPSILON)

    # 9. CCCI: NDRE / NDVI
    ccci = ndre / (ndvi + EPSILON)  # Add EPSILON to avoid zero division
    
    # --- Compute means for each index across the array ---
    result = {
        'NDVI': ndvi.mean().item(),
        'EVI': evi.mean().item(),
        'NDWI': ndwi.mean().item(),
        'GNDVI': gndvi.mean().item(),
        'SAVI': savi.mean().item(),
        'MSAVI': msavi.mean().item(),
        'MoistureIndex': moisture_idx.mean().item(),
        'NDRE': ndre.mean().item(),
        'CCCI': ccci.mean().item()
    }

    # Close the data to free memory
    data_array.close()
    
    return result

def compute_indices_for_df(final_trdf, tif_path_col='tif_path'):
    """
    For each row in `final_trdf`, read the TIF file path,
    compute vegetation indices, and store the results as new columns.
    """
    # List or dictionary to store results
    # We'll append directly to the dataframe for convenience
    index_columns = [
        'NDVI', 'EVI', 'NDWI', 'GNDVI', 'SAVI', 'MSAVI', 
        'MoistureIndex', 'NDRE', 'CCCI'
    ]

    # Initialize columns to NaN in case of errors or missing files
    for col in index_columns:
        final_trdf[col] = np.nan

    # Iterate over rows to compute and assign index values
    for idx, row in final_trdf.iterrows():
        tif_path = row[tif_path_col]
        if not isinstance(tif_path, str):
            print(f"Skipping index {idx} due to invalid path: {tif_path}")
            continue

        try:
            # Compute the indices
            result = compute_indices(tif_path)

            # Assign values to dataframe columns
            for k, v in result.items():
                final_trdf.at[idx, k] = v

        except Exception as e:
            print(f"Error processing row {idx} with TIF path {tif_path}: {e}")

    return final_trdf


def prepare_model_data(df, type='train'):
    """
    Prepares data for machine learning by encoding categorical columns
    according to specified rules:
    - For 'train': 'category' is mapped to 'target' using label encoding.
    - 'Crop' is one-hot or label encoded.
    - 'CNext' and 'CLast' share the same encoding to maintain consistency.
    - 'CTransp', 'IrriType', 'IrriSource', and 'Season' are label encoded.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to process.
    type : str, optional (default='train')
        The type of data being processed ('train' or 'test').

    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame ready for model training or testing.
    """

    # Handle 'category' only if type is 'train'
    if type == 'train' and 'category' in df.columns:
        # Encode 'category' to 'target'
        category_mapper = {label: idx for idx, label in enumerate(df['category'].unique()) if pd.notna(label)}
        df['target'] = df['category'].map(category_mapper)
        # Drop the original 'category' column now that we have 'target'
        df.drop(columns=['category'], inplace=True)

    # One-hot encoding or label encoding for 'Crop'
    if 'Crop' in df.columns:
        encoder = LabelEncoder()
        df['Crop'] = encoder.fit_transform(df['Crop'])

    # Consistent Label Encoding for 'CNext' and 'CLast'
    if 'CNext' in df.columns and 'CLast' in df.columns:
        combined_values = pd.concat([df['CNext'], df['CLast']]).unique()
        cl_encoder = LabelEncoder()
        cl_encoder.fit(combined_values)

        # Apply shared encoding to both columns
        df['CNext'] = cl_encoder.transform(df['CNext'])
        df['CLast'] = cl_encoder.transform(df['CLast'])

    # Label Encode other columns
    label_cols = ['CTransp', 'IrriType', 'IrriSource', 'Season']
    for col in label_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    return df