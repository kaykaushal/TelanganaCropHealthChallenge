import os
import ee
import geemap
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from shapely.affinity import scale, translate
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Utility Functions
# ----------------------------

def format_date(date_val):
    """
    Convert a date to 'YYYY-MM-DD' format.
    Handles string dates in various formats or pandas Timestamps.
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
    """
    centroid = geometry.centroid
    translated_geometry = translate(geometry, -centroid.x, -centroid.y)
    scaled_geometry = scale(translated_geometry, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return translate(scaled_geometry, centroid.x, centroid.y)


def generate_monthly_date_ranges(start_date, end_date):
    """
    Generate a list of monthly date ranges between start_date and end_date.
    """
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_ranges = []

    while current_date <= end_date:
        next_date = current_date + relativedelta(months=1) - timedelta(days=1)
        next_date = min(next_date, end_date)  # Ensure it doesn't exceed end_date
        date_ranges.append((current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current_date += relativedelta(months=1)

    return date_ranges


def download_monthly_mean_image(collection_name, bands, start_date, end_date, region, output_folder, month_folder, farm_id):
    """
    Download monthly mean image from a GEE collection filtered by date and region.
    Ensure unique images per FarmID.
    """
    collection = ee.ImageCollection(collection_name).filterDate(start_date, end_date).filterBounds(region)
    image = collection.sort('system:time_start', False).first().select(bands).clip(region)

    # Ensure FarmID is included in the filename for uniqueness
    image_name = f"{output_folder}/{month_folder}/Farm_{farm_id}_RGB_{start_date}_{end_date}.tif"

    # Check if the file already exists to prevent duplicate downloads
    if os.path.exists(image_name):
        print(f"File already exists for FarmID {farm_id}, skipping download: {image_name}")
        return image_name

    # Create the monthly folder if it doesn't exist
    os.makedirs(f"{output_folder}/{month_folder}", exist_ok=True)

    # Download the mean image
    geemap.ee_export_image(
        image,
        filename=image_name,
        scale=10,
        region=region,
        file_per_band=False,
        crs="EPSG:4326"
    )
    print(f"Downloaded: {image_name}")
    return image_name


def process_row_monthly(index, row, output_folder, start_date_col, end_date_col):
    """
    Process a single row from the DataFrame to download Sentinel-2 monthly images.
    Ensures unique images per FarmID and stores TIFF paths correctly.
    """
    start_date = format_date(row[start_date_col])
    end_date = format_date(row[end_date_col])
    farm_id = row["FarmID"]  # Get unique FarmID

    if not start_date or not end_date:
        print(f"Skipping entry due to invalid dates: {start_date_col}={row[start_date_col]}, {end_date_col}={row[end_date_col]}")
        return index, row

    region_geometry = row["geometry"]

    if region_geometry.is_empty:
        print(f"Skipping empty geometry for row {index}")
        return index, row

    if region_geometry.geom_type == "Polygon":
        region = ee.Geometry.Polygon(region_geometry.__geo_interface__["coordinates"])
    elif region_geometry.geom_type == "MultiPolygon":
        coords = [polygon.exterior.coords[:] for polygon in region_geometry.geoms]
        region = ee.Geometry.MultiPolygon(coords)
    else:
        print(f"Unsupported geometry type for row {index}: {region_geometry.geom_type}")
        return index, row

    try:
        image_collection_name = "COPERNICUS/S2"
        bands = ["B4", "B3", "B2", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]

        # Generate monthly date ranges
        monthly_date_ranges = generate_monthly_date_ranges(start_date, end_date)

        for i, (month_start, month_end) in enumerate(monthly_date_ranges, start=1):
            month_folder = f"M{i}"
            image_path = download_monthly_mean_image(
                image_collection_name, bands, month_start, month_end, region, output_folder, month_folder, farm_id
            )
            row[f"M{i}_tif_path"] = image_path

        return index, row
    except Exception as e:
        print(f"Error processing entry for row {index}: {e}")
        return index, row


def download_s2_monthly(df, output_folder, start_date_col, end_date_col, filename, scale_factor, download=True):
    """
    Main function to download Sentinel-2 monthly images based on input data.
    Saves the resulting DataFrame with columns for each month's TIFF path.
    """
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project="ee-kaykishore5")

    # Scale geometries
    df["geometry"] = df["geometry"].apply(scale_geometry, scale_factor=scale_factor)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    if download:
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row_monthly, index, row, output_folder, start_date_col, end_date_col)
                       for index, row in df.iterrows()]
            for future in as_completed(futures):
                index, updated_row = future.result()
                results.append(updated_row)
        
        updated_df = pd.DataFrame(results)
        updated_df.to_csv(f"{output_folder}/{filename}", index=False)
        print(f"Saved updated DataFrame to {output_folder}/{filename}")
    else:
        data_path = f"{output_folder}/{filename}"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"The file {data_path} does not exist. Please set download=True.")
    print("Process completed!")