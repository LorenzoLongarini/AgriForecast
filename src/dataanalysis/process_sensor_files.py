import os
import pandas as pd
from glob import glob

def process_sensor_files(input_folder, output_folder):
    """
    Process all CSV files in the input folder, grouping data by sensor ID, and
    save a single file for each sensor with the complete temporal data.

    Args:
        input_folder (str): Path to the folder containing the CSV files.
        output_folder (str): Path to the folder where output files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Find all CSV files in the input folder
    file_paths = glob(os.path.join(input_folder, "*.csv"))

    if not file_paths:
        print("No CSV files found in the specified folder.")
        return

    # Initialize a dictionary to hold data for each sensor
    sensor_data = {}

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Read the CSV file, assuming a semicolon separator
        df = pd.read_csv(file_path, sep=';')
        print(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")

        # Ensure timestamp or datetime is parsed correctly
        if 'datetimeV' in df.columns:
            df['datetimeV'] = pd.to_datetime(df['datetimeV'])
            print(f"datetimeV column converted to datetime")

        # Group data by sensor ID and append to the dictionary
        for sensor_id, group in df.groupby('id'):
            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = []
            sensor_data[sensor_id].append(group)
            print(f"Appended data for sensor ID {sensor_id}")

    # Combine and save data for each sensor
    for sensor_id, data_list in sensor_data.items():
        # Concatenate all dataframes for this sensor
        combined_df = pd.concat(data_list).sort_values(by='datetimeV')
        print(f"Combined DataFrame for sensor ID {sensor_id} with {len(combined_df)} rows")

        # Save to a CSV file
        output_file = os.path.join(output_folder, f"{sensor_id}.csv")
        combined_df.to_csv(output_file, index=False, sep=';')
        print(f"Data for sensor {sensor_id} saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_folder = r".\\assets\\dativalorizzati"  # Replace with the path to your input folder
    output_folder = r".\\assets\\datipersensore"  # Replace with the path to your output folder

    process_sensor_files(input_folder, output_folder)