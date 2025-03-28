import os
import csv

root_dir = r"Z:\BZLab2\Kelvin\EM"
output_csv = "EM_metadata.csv"

metadata = []

for pathology in ['ASD', 'Control']:
    pathology_dir = os.path.join(root_dir, pathology)
    if not os.path.isdir(pathology_dir):
        continue

    for patient_id in os.listdir(pathology_dir):
        patient_path = os.path.join(pathology_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        print(patient_path)
        for region_depth_folder in os.listdir(patient_path):
            rd_path = os.path.join(patient_path, region_depth_folder)
            if not os.path.isdir(rd_path):
                continue
            print(rd_path)
            # Extract region and depth from folder name
            region = None
            depth = None
            for candidate_region in ['A25', 'A46', 'OFC']:
                if candidate_region in region_depth_folder:
                    region = candidate_region
                    break
            for candidate_depth in ['DWM', 'SWM']:
                if candidate_depth in region_depth_folder:
                    depth = candidate_depth
                    break
                
            if region is None or depth is None:
                continue  # Skip folders that don't match naming pattern

            for root, _, files in os.walk(rd_path):
                for file in files:
                    if file.lower().endswith('.tif'):
                        file_path = os.path.join(root, file)
                        metadata.append({
                            'pathology': pathology,
                            'patient_id': patient_id,
                            'region': region,
                            'depth': depth,
                            'filepath': file_path
                        })

# Write to CSV
with open(output_csv, mode='w', newline='') as csv_file:
    fieldnames = ['pathology', 'patient_id', 'region', 'depth', 'filepath']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for entry in metadata:
        writer.writerow(entry)

print(f"Metadata CSV has been written to {output_csv}")