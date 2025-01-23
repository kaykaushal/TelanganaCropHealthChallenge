import os
import ee
import geemap
import pandas as pd
from datetime import datetime
from shapely.affinity import scale, translate
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Utility Functions
# ----------------------------

def format_date(date_val):
    """
    Convert a date to 'YYYY-MM-DD' format.
    Handles string dates in various formats or pandas Timestamps.

    Parameters:
        date_val (str | pd.Timestamp): The date to format.

    Returns:
        str | None: Formatted date as 'YYYY-MM-DD' or None if invalid.
    """
    if isinstance(date_val, pd.Timestamp):
        return date_val.strftime('%Y-%m-%d')

    if isinstance(date_val, str):
        accepted_formats = ['%d-%m-%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S']
        for fmt in accepted_formats:
            try:
                return datetime.strptime(date_val.split()[0], fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue

    print(f"Error: Invalid date format for {date_val}.")
    return None


def scale_geometry(geometry, scale_factor):
    """
    Scale a geometry object around its centroid.

    Parameters:
        geometry (shapely.geometry.base.BaseGeometry): Geometry to scale.
        scale_factor (float): Factor by which to scale the geometry.

    Returns:
        shapely.geometry.base.BaseGeometry: Scaled geometry object.
    """
    centroid = geometry.centroid
    translated_geometry = translate(geometry, -centroid.x, -centroid.y)
    scaled_geometry = scale(translated_geometry, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return translate(scaled_geometry, centroid.x, centroid.y)


def download_rgb_image(collection_name, bands, start_date, end_date, region, output_folder):
    """
    Download RGB bands from a GEE collection filtered by date and region.

    Parameters:
        collection_name (str): Name of the Earth Engine collection.
        bands (list): List of band names to select.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        region (ee.Geometry): Region to clip the image.
        output_folder (str): Folder to save the downloaded image.

    Returns:
        str: Path to the saved image.
    """
    collection = ee.ImageCollection(collection_name).filterDate(start_date, end_date).filterBounds(region)
    image = collection.sort('system:time_start', False).first().select(bands).clip(region)
    image_id = image.id().getInfo() or f'image_{start_date}_{end_date}'
    image_name = f'{output_folder}/{image_id}_RGB_{start_date}_{end_date}.tif'

    # Check if the file already exists
    if os.path.exists(image_name):
        print(f"File already exists, skipping download: {image_name}")
        return image_name

    # Download the image
    geemap.ee_export_image(
        image,
        filename=image_name,
        scale=10,
        region=region,
        file_per_band=False,
        crs='EPSG:4326'
    )
    print(f"Downloaded: {image_name}")
    return image_name


def process_row(index, row, output_folder, start_date_col, end_date_col):
    """
    Process a single row from the DataFrame to download Sentinel-2 images.

    Parameters:
        index (int): Row index.
        row (pd.Series): Row data.
        output_folder (str): Folder to save the downloaded image.
        start_date_col (str): Name of the column with start dates.
        end_date_col (str): Name of the column with end dates.

    Returns:
        tuple: Index and downloaded image path.
    """
    start_date = format_date(row[start_date_col])
    end_date = format_date(row[end_date_col])

    if not start_date or not end_date:
        print(f"Skipping entry due to invalid dates: {start_date_col}={row[start_date_col]}, {end_date_col}={row[end_date_col]}")
        return index, None

    region_geometry = row['geometry']

    if region_geometry.is_empty:
        print(f"Skipping empty geometry for row {index}")
        return index, None

    if region_geometry.geom_type == 'Polygon':
        region = ee.Geometry.Polygon(region_geometry.__geo_interface__['coordinates'])
    elif region_geometry.geom_type == 'MultiPolygon':
        coords = [polygon.exterior.coords[:] for polygon in region_geometry.geoms]
        region = ee.Geometry.MultiPolygon(coords)
    else:
        print(f"Unsupported geometry type for row {index}: {region_geometry.geom_type}")
        return index, None

    try:
        image_collection_name = 'COPERNICUS/S2'
        bands = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        image_file = download_rgb_image(image_collection_name, bands, start_date, end_date, region, output_folder)
        return index, image_file
    except Exception as e:
        print(f"Error processing entry for row {index}: {e}")
        return index, None


def main(df, output_folder, start_date_col, end_date_col, filename, scale_factor, download=True):
    """
    Main function to download Sentinel-2 images based on input data.

    Parameters:
        df (pd.DataFrame): Input DataFrame with geometry and date columns.
        output_folder (str): Folder to save the downloaded images.
        start_date_col (str): Name of the start date column.
        end_date_col (str): Name of the end date column.
        filename (str): Name of the output CSV file.
        download (bool): Whether to download images or load existing paths.
        scale_factor (float): Scaling factor for geometries.
    """
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project="ee-kaykishore5")

    # Scale geometries
    df['geometry'] = df['geometry'].apply(scale_geometry, scale_factor=scale_factor)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    if download:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, index, row, output_folder, start_date_col, end_date_col)
                       for index, row in df.iterrows()]
            for future in as_completed(futures):
                index, image_file = future.result()
                if image_file:
                    df.at[index, 'tif_path'] = image_file
        df.to_csv(f"{output_folder}/{filename}", index=False)
    else:
        data_path = f"{output_folder}/{filename}"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"The file {data_path} does not exist. Please set download=True.")
    print("Process completed!")