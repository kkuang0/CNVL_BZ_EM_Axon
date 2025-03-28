from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class AxonDataset(Dataset):
    def __init__(self, metadata_df, base_dir, transform=None, augment=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image metadata
            base_dir (str): Base directory containing the Images folder
            transform (callable, optional): Base transforms
            augment (callable, optional): Augmentation transforms
        """
        self.metadata = metadata_df
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.augment = augment
        
        # Create label encoders
        self.pathology_map = {'Control': 0, 'ASD': 1}
        self.region_map = {'A25': 0, 'A46': 1, 'OFC': 2}
        self.depth_map = {'DWM': 0, 'SWM': 1}
        
    def __len__(self):
        """
        Required for DataLoader to know the size of the dataset.
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image path and labels
        row = self.metadata.iloc[idx]

        # Load image
        image = Image.open(row['filepath']).convert('L')  # Convert to grayscale
 
        
        # Apply augmentation if specified
        if self.augment is not None:
            image = self.augment(image)
        
        # Apply base transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels
        pathology_label = self.pathology_map[row['pathology']]
        region_label = self.region_map[row['region']]
        depth_label = self.depth_map[row['depth']]
        
        return {
            'image': image,
            'pathology': pathology_label,
            'region': region_label,
            'depth': depth_label,
            'patient_id': row['patient_id']
        }