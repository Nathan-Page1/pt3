import os
import numpy as np
import matplotlib.pyplot as plt
from BioMSFA_utils import (
    BioMSFA_raw_im_to_npy,
    pixspec_BioMSFA_norm,  # Pre-loaded calibration data for normalization
    gaussian_weights,
    BioMSFA_filtermask,
    generate_mask_triangulations,
    interpolate_from_triangulations,
    cube_recombine
)
from General_utils import norm_by_percentile

# Define the workspace and subfolders
workspace = "2024-11-05_BioMSFA_USAF_Test"
dark_folder = os.path.join(workspace, "Dark Images")
test_folder = os.path.join(workspace, "Test Images")
white_folder = os.path.join(workspace, "White Images")

def load_images_from_folder(folder):
    """Load all images in a given folder and return as a list of arrays."""
    images = []
    for filename in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        if filename.endswith(".raw"):
            image = BioMSFA_raw_im_to_npy(file_path)
            images.append(image)
    return np.array(images)

def average_images(images):
    """Average a stack of images."""
    return np.mean(images, axis=0)

def normalize_with_calibration(test_image, calibration_data):
    """Normalize the test image using calibration data from BioMSFA_utils."""
    normalized_image = (test_image - calibration_data["dark"]) / (calibration_data["white"] - calibration_data["dark"])
    return norm_by_percentile(normalized_image, 99.9)

def calibrate_image(image):
    """Calibrate the image using BioMSFA filter masks and triangulations for interpolation."""
    triangulations = generate_mask_triangulations(BioMSFA_filtermask)
    calibrated_image = interpolate_from_triangulations(image, BioMSFA_filtermask, triangulations)
    return calibrated_image

def form_3d_array(image):
    """Form a 3D image cube using calibration weights."""
    image_cube = cube_recombine(image, weights=gaussian_weights)
    return image_cube

def calculate_mtf(image):
    """Calculate the Modulation Transfer Function (MTF) from the image."""
    ft_image = np.fft.fftshift(np.fft.fft2(image))
    mtf = np.abs(ft_image)
    return mtf / np.max(mtf)

def display_mtf(mtf, resolution_cutoff=0.1):
    """Display MTF and resolution cutoff."""
    plt.imshow(mtf, cmap="viridis")
    plt.colorbar(label="MTF")
    plt.title("Modulation Transfer Function (MTF)")
    plt.show()

    # Display resolution cutoff as contour
    plt.contour(mtf, levels=[resolution_cutoff], colors='red')
    plt.title(f"Resolution Cutoff at {resolution_cutoff}")
    plt.show()

# Main Analysis Pipeline
def main_analysis_pipeline():
    # Step 1: Load and average images from each folder
    dark_images = load_images_from_folder(dark_folder)
    test_images = load_images_from_folder(test_folder)
    white_images = load_images_from_folder(white_folder)

    # Step 2: Average images in each folder
    dark_image_avg = average_images(dark_images)
    white_image_avg = average_images(white_images)

    # Create calibration data dictionary for normalization
    calibration_data = {
        "dark": dark_image_avg,
        "white": white_image_avg,
        "pixspec_norm": pixspec_BioMSFA_norm  # Already normalized spectra from BioMSFA_utils
    }

    # Process each test image
    for i, test_image in enumerate(test_images):
        # Step 3: Normalize using calibration data
        normalized_image = normalize_with_calibration(test_image, calibration_data)

        # Step 4: Calibrate the image
        calibrated_image = calibrate_image(normalized_image)

        # Step 5: Form a 3D image cube
        image_cube = form_3d_array(calibrated_image)

        # Step 6: Calculate MTF from the normalized or calibrated image
        mtf = calculate_mtf(normalized_image)

        # Step 7: Display the MTF and resolution
        print(f"Displaying MTF for test image {i + 1}")
        display_mtf(mtf)

# Run the analysis pipeline
if __name__ == "__main__":
    main_analysis_pipeline()
