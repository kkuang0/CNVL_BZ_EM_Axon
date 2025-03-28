import random
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image, ImageFilter, ImageEnhance

class AxonAugmenter:
    """Custom augmentation pipeline for axon microscopy images."""
    
    def __init__(self,
                 rotation_range=360,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 noise_range=(0.0, 0.05),
                 blur_prob=0.3,
                 blur_radius=(0.5, 1.5),
                 zoom_range=(0.85, 1.15),
                 flip_prob=0.5):
        """
        Args:
            rotation_range (int): Range for random rotation
            brightness_range (tuple): Range for brightness adjustment
            contrast_range (tuple): Range for contrast adjustment
            noise_range (tuple): Range for noise amplitude
            blur_prob (float): Probability of applying Gaussian blur
            blur_radius (tuple): Range for blur radius
            zoom_range (tuple): Range for random zoom
            flip_prob (float): Probability of horizontal/vertical flip
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_range = noise_range
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        self.zoom_range = zoom_range
        self.flip_prob = flip_prob
    
    def add_microscope_noise(self, image):
        """Add realistic microscope noise (Gaussian + Poisson)."""
        image_array = np.array(image)
        
        # Add Gaussian noise
        noise_amplitude = random.uniform(*self.noise_range)
        gaussian_noise = np.random.normal(0, noise_amplitude, image_array.shape)
        
        # Add Poisson noise (simulate photon counting noise)
        poisson_noise = np.random.poisson(image_array).astype(float) - image_array
        poisson_noise *= noise_amplitude
        
        # Combine noises and clip to valid range
        noisy_image = image_array + gaussian_noise + poisson_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)
    
    def random_blur(self, image):
        """Apply random Gaussian blur."""
        if random.random() < self.blur_prob:
            radius = random.uniform(*self.blur_radius)
            return image.filter(ImageFilter.GaussianBlur(radius))
        return image
    
    def random_zoom(self, image):
        """Apply random zoom while maintaining aspect ratio."""
        zoom_factor = random.uniform(*self.zoom_range)
        
        # Calculate new dimensions
        w, h = image.size
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Resize and crop/pad to original size
        if zoom_factor > 1:  # Zoom in
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
        else:  # Zoom out
            temp_image = Image.new(image.mode, (w, h), (0,))
            resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            temp_image.paste(resized, (left, top))
            image = temp_image
            
        return image
    
    def __call__(self, image):
        """Apply the augmentation pipeline to an image."""
        # Convert to PIL if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Basic geometric transformations
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range/2, self.rotation_range/2)
            image = image.rotate(angle, Image.Resampling.BILINEAR, expand=False)
        
        # Intensity transformations
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(*self.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = random.uniform(*self.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        # Microscope-specific augmentations
        image = self.random_blur(image)
        image = self.add_microscope_noise(image)
        image = self.random_zoom(image)
        
        return image