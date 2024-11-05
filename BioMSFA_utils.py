"""
Utility file for BioMSFA analysis, created on 14th August 2024.
A previous version of this code exists under the name BioMSFA_utils_preAug24, which contains much
of the below code plus some snippets which are now obsolete. Some code was moved into the General_utils.py
file (previously just called 'config') while the code below is now structured to better reflect the pipeline.
"""

# %% Generically useful packages & functions
import numpy as np
import pandas as pd

# %% Data Paths & File Names
from General_utils import root_directory
BioMSFA_DataPath = root_directory + "Calum MSFA/"

# %% Design-based Data
# Location and Dimensions of BioMSFA patterned region relative to full sensor
BioMSFA_locs= [515,1861,882,2973] # [y0,y1,x0,x1] ~ this needs to be consistent between analyses!
BioMSFA_shape=[BioMSFA_locs[1]-BioMSFA_locs[0],BioMSFA_locs[3]-BioMSFA_locs[2]]
BioMSFA_adjlocs = [531,1877,896,2987] # y-vals + 16 , x-vals + 14 to account for transform to (3872x2192) image format
# Dose Test Patches (above BioMSFA Region)
testpatches = pd.DataFrame(columns = ["Dose","x0","y0","x1","y1"])
testpatches.Dose = [250, 272, 296, 323, 352, 383, 417,454,494,539,586,639,695,757,825,898,978,1065,1180,1263,1376,1498, 1631]
testpatches.x0 = [3009,2907,2808,2704,2606,2508,2402,2304,2204,2110,2010,1904,1807,1706,1606,1506,1407,1305,1206,1106,1005,907,805]
testpatches.x1 = testpatches["x0"]+40
testpatches.y0 = 237
testpatches.y1 = 277

# %% Calibration Data
calibration_parent_folder = BioMSFA_DataPath+"2023 Fianium Sweep/"
calibration_dataset = "20231027-145800_SplitPlusCombined_BioMSFA3_400.0nm-900.0nm_d5.0"
calibration_datafolder = calibration_parent_folder + calibration_dataset + "/Output Data/"
# Wavelengths used for calibration
calibration_wavelengths= np.load(calibration_datafolder+"wavelengths.npy")
# Array of pixel spectra across sensor, cropped to BioMSFA region
pixspec_full = np.load(calibration_datafolder+"3d_pixel_spectra_image.npy")
pixspec_BioMSFA = pixspec_full[:,BioMSFA_locs[0]:BioMSFA_locs[1],BioMSFA_locs[2]:BioMSFA_locs[3]]
from General_utils import rolling_mean_smoothing1d,norm_by_percentile
pixspec_BioMSFA_smooth = rolling_mean_smoothing1d(pixspec_BioMSFA,window_width=10,axis=0)
pixspec_BioMSFA_norm = norm_by_percentile(pixspec_BioMSFA_smooth,99.99)
# Masks based on groupings of pixels of similar response spectra
BioMSFA_filtermask = np.load(calibration_datafolder+"mask_sam0.4_lowtol-designmasked-endmembers_mask.npy")
BioMSFA_maskspecs = np.load(calibration_datafolder+"mask_sam0.4_lowtol-designmasked-endmembers_maskspectra_mean.npy")
BioMSFA_maskspecstds = np.load(calibration_datafolder+"mask_sam0.4_lowtol-designmasked-endmembers_maskspectra_stds.npy")

# %% Image Demosaicking & Interpolation
def BioMSFA_raw_im_to_npy(file,imshape=(2192, 3872),crop=True,cropshape=BioMSFA_adjlocs):
    im = np.fromfile(file, dtype=np.uint16).byteswap(inplace=True).reshape(imshape)
    if crop==True:
        im = im[cropshape[0]:cropshape[1],cropshape[2]:cropshape[3]]
    return im
def raw_im_to_npy_mean(raw_file_name_list,imshape=(2192, 3872),asint=True,troubleshoot=False,name="Contributing Images"):
    hist = []
    imstack = []
    count=0
    for file in raw_file_name_list:
        im = np.fromfile(file, dtype=np.uint16).byteswap(inplace=True).reshape(imshape)
        if troubleshoot==True:
            plt.hist(im.flatten(),bins=500,alpha=0.1,label=count)
            count=count+1
        imstack.append(im)
    imstack = np.array(imstack)
    meanim = imstack.mean(axis=0)
    if troubleshoot==True:
        plt.hist(meanim.flatten(),bins=500,alpha=0.2,label="Mean float64")
    if asint==True:
        meanim=meanim.astype(np.uint16)
    if troubleshoot==True:
        plt.hist(meanim.flatten(),bins=500,alpha=0.2,label="Mean uint16")
        plt.title(name)
        plt.xlim(0,65536)
        plt.legend(ncols=int(np.sqrt(len(raw_file_name_list)+2)))
        plt.show()
    return meanim

# Interpolation
def generate_mask_triangulations(mask):
    """
    :param mask: numpy array of mask layers, binary 1=True,0=False, axes: 0-mask layers, 1-y, 2-x.
    :return: triangulations for interpolation between
    """
    from scipy.spatial import Delaunay
    # Triangulation weightings for interpolation
    triangulations = []
    for f in range(0,mask.shape[0]):
        # mask locations
        x_mask = np.nonzero(mask[f])[1]
        y_mask = np.nonzero(mask[f])[0]
        triangulations.append(Delaunay(list(zip(x_mask,y_mask))))
    return triangulations
def interpolate_from_triangulations(image,mask,triangulations):
    from scipy.interpolate import LinearNDInterpolator
    if (mask.shape[1]!=image.shape[0] or mask.shape[2]!=image.shape[1]):
        print("mask and image shapes are not correctly aligned")
    x_out = np.arange(0, image.shape[1], 1)
    y_out = np.arange(0, image.shape[0], 1)
    x_out, y_out = np.meshgrid(x_out, y_out)
    interpolated_image = []
    # Now re-use the precalculated triangulations to interpolate masked images
    for f in range(0, mask.shape[0]):
        z_vals = image[mask[f]]
        gridinterp = LinearNDInterpolator(triangulations[f], z_vals)
        interpolated_image.append(gridinterp(x_out,y_out))
    interpolated_image = np.array(interpolated_image)
    return interpolated_image

BioMSFA_triangulations =generate_mask_triangulations(mask=BioMSFA_filtermask)


# %% Multispectral Image Cube Recombination
from General_utils import gaussian
# Centres, std devs and recombinatino weights of gaussians optimised for minimal difference of reconstructed spectra
gaussian_centres = np.array([463, 536, 563, 586, 651, 692, 773, 809, 840])
gaussian_sigmas = np.array([20, 20, 15, 20, 30, 30, 25, 15, 15])
gaussians = np.array([gaussian(calibration_wavelengths,gaussian_centres[i],gaussian_sigmas[i]) for i in range(len(gaussian_sigmas))]).T
gaussian_weights = np.matmul(np.linalg.pinv(BioMSFA_maskspecs.T),gaussians).T
# RGB Image Reconstruction Weights
from scipy.ndimage.filters import gaussian_filter
rgb_pd = pd.read_csv("/Users/katie-lou/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/Data/Internet datasets/boazarad_NTIRE2022_spectral_RGB_Camera_QE.csv")
rgb_pd = rgb_pd.set_index("Wavelength[nm]")
[r,g,b] = gaussian_filter([np.interp(calibration_wavelengths,rgb_pd.index,rgb_pd.R),
                           np.interp(calibration_wavelengths,rgb_pd.index,rgb_pd.G1),
                           np.interp(calibration_wavelengths,rgb_pd.index,rgb_pd.B)],
                          sigma=(0,2))
rgb_targets = np.array([r,g,b])
rgb_targets = rgb_targets/rgb_targets.max() # Normalise to max transmission = 1
RGB_weights = np.matmul(np.linalg.pinv(BioMSFA_maskspecs.T),rgb_targets.T).T
def cube_recombine(imcube,weights=gaussian_weights):
    gaussim = np.rollaxis(np.matmul(weights,np.rollaxis(imcube,0,2)),0,2)
    gaussim = gaussim*(gaussim>0)
    return gaussim

# %% Small Region Spectral Unmixing
# Hyperspectral Regions
import math
ysf = (1861-515)/(2688)
xsf = (2973-882)/(4176)
hslocs = pd.DataFrame(columns=["x0","x1","y0","y1"])
hslocs.loc[len(hslocs)]=[2040,2112,1416,1488]
hslocs.loc[len(hslocs)]=[3384,3456,2376,2448]
hslocs.loc[len(hslocs)]=[3408,3432,408,	432]
hslocs.loc[len(hslocs)]=[2928,2952,1512,1536]
hslocs.loc[len(hslocs)]=[2112,2136,384,408]
hslocs.loc[len(hslocs)]=[2088,2112,408,432]
hslocs.loc[len(hslocs)]=[2064,2088,384,408]
hslocs.loc[len(hslocs)]=[2112,2136,432,456]
hslocs.loc[len(hslocs)]=[2064,2088,432,456]
hslocs.loc[len(hslocs)]=[552,576,408,432]
hslocs.loc[len(hslocs)]=[1224,1248,1512,1536]
hslocs.loc[len(hslocs)]=[720,792,2376,2448]
hslocs.loc[len(hslocs)]=[2064,2088,2400,2424]
hslocs_scaled = hslocs.copy()
hslocs_scaled.x0 = hslocs["x0"].apply(lambda x: math.floor(x*xsf))
hslocs_scaled.x1 = hslocs["x1"].apply(lambda x: math.ceil(x*xsf))
hslocs_scaled.y0 = hslocs["y0"].apply(lambda y: math.floor(y*ysf))
hslocs_scaled.y1 = hslocs["y1"].apply(lambda y: math.ceil(y*ysf))
hslocs_scaled.loc[11]=[263,299,1190,1226] # Correction after plotting image. Kept original above just in case.
hslocs_scaled["label"]=[5,9,3,6,2,2,2,2,2,1,4,7,8] # Add label
# List of pixelspectra in Hyperspectral Regions, retaining dimensions
hs_pixspecs = []
for i in range(0,len(hslocs_scaled)):
    [x0,x1,y0,y1]=hslocs_scaled.iloc[i][["x0","x1","y0","y1"]].to_numpy()
    hs_pixspecs.append(pixspec_BioMSFA_smooth[:,y0:y1,x0:x1])
hslocs_scaled["pixspecs"]=hs_pixspecs
pd.to_pickle(hslocs_scaled,root_directory+"/Calum MSFA/hs locations - scaled sensor perspective.pkl")
# ADAM optimiser for finding spectra from pixel signals
import torch
from torch.functional import F
import torch.nn as nn
from tqdm import tqdm
class PixvalSpectrumOptimizer(nn.Module):
    """Pytorch model for custom gradient optimization of constituent spectra concentrations."""
    def __init__(self, wavelengths, nr_of_endmembers, filters, lr=0.01,n_iter=10000):
        super().__init__()
        self.nr_of_endmembers = nr_of_endmembers
        self.wavelengths = wavelengths
        # Initialise with random weights
        e_weights = torch.distributions.Uniform(0, 1).sample((nr_of_endmembers,))
        # make weights torch parameters
        self.endmember_weights = nn.Parameter(e_weights)
        self.n_iter = n_iter
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.transmission_spectra = filters

    def forward(self, X):
        mixed_spectrum = torch.matmul(self.endmember_weights, X)
        measured_filter_values = torch.matmul(mixed_spectrum, self.transmission_spectra.T)
        return measured_filter_values

    def loss(self, prediction, target):
        absolute_difference = F.mse_loss(prediction, target)
        # could add a step function: if negative values in prediction,
        return absolute_difference

    def train_loop(self, input_spectra, target_spectrum):
        """optimizing loop for minimizing given loss term.

        :param input_spectra: spectra of possible endmembers
        :param target_spectrum:
        :return: list of losses
        """
        losses = []
        for i in (pbar := tqdm(range(self.n_iter))):
            preds = self.forward(input_spectra)
            loss = self.loss(preds, target_spectrum)
            # loss += 0.5*torch.sum(self.concentrations)
            loss.backward()
            self.optimizer.step()
            self.endmember_weights.data = self.endmember_weights.data.clamp(min=0)
            self.optimizer.zero_grad()
            losses.append(loss)
            pbar.set_description(f"loss = {loss:.4f}")
        return losses
def opti_unmix_pix_with_loss(endmembers,wavelengths,pixvals,transmission,lr=0.001,n_iter=10000):
# Initialise optimiser
    optim = PixvalSpectrumOptimizer(torch.from_numpy(wavelengths).type(torch.float32),  # wavelengths
                                endmembers.shape[0],  # nr_endmembers
                                filters=torch.from_numpy(transmission).type(torch.float32),
                                lr=lr, n_iter=n_iter)# filters
    losses = optim.train_loop(torch.from_numpy(endmembers).type(torch.float32),
                          torch.from_numpy(pixvals).type(torch.float32))
    weights = optim.endmember_weights.detach().numpy()
    lossarray = [losses[i].detach().numpy() for i in range(len(losses))]
    return weights,lossarray
# Specific functions for getting spectra from hyperspectral regions
def interp_downsample(array,start_ds=400,end_ds=900,n_ds=20):
    # Only works when the
    interparray = np.interp(np.linspace(start_ds,end_ds,n_ds),np.linspace(np.nanmin(array),np.nanmax(array),np.count_nonzero(~np.isnan(array))),array)
    return interparray
def downsample_hs(hs_ds,wavelengths,illumination,n_ds=20,n_k=20):
    from sklearn.cluster import KMeans
    wavs_ds = interp_downsample(wavelengths)
    ill_ds = interp_downsample(illumination)
    # Create empty arrays for downsampled transmission spectra of grouped pixels
    # T_ds = downsampled raw spectra
    # Till_ds = downsampled illumination modulated spectra
    # Tk_group = locations of pixels which have been grouped by KMeans clustering based on spectral shape
    T_ds, Till_ds, Tk_group = [], [], []
    for h in hs_ds.index:
        Th = hs_ds.iloc[h].pixspecs
        Th = Th.reshape(len(wavelengths), Th.shape[1] * Th.shape[2])
        Thflat = np.array([interp_downsample(Th[:, i]) for i in range(Th.shape[1])]).T
        T_ds.append(Thflat)
        Till_ds.append(Thflat * np.repeat(ill_ds[:, np.newaxis], Thflat.shape[1], axis=1))
        # kmeans clustering
        Tk_group.append(KMeans(n_clusters=n_k, n_init="auto", max_iter=1000).fit((Thflat / Thflat.max(axis=0)).T).labels_)
    hs_ds["T_ds"] = T_ds
    hs_ds["Till_ds"] = Till_ds
    hs_ds["Tk_group"] = Tk_group
    return wavs_ds,hs_ds,ill_ds
def hs_unmix(image,illumination, wavelengths=calibration_wavelengths,n_ds=20, n_k=20,grouped=False, illmod=True,lr=0.00005, n_iter=40000):
    """
    :param image:           Raw image being processed
    :param illumination:    Illumination spectral profile
    :param wavelengths:     Calibration wavelength basis set
    :param n_ds:            Number of samples in downsampled set
    :param n_k:             Number of k-means groups
    :param grouped:         Boolean determiner. True: Use k-means clusters to group similar pixels, False: Use all pixels individually
    :param illmod:          Boolean determiner. True: Use illumination modulated pixel transmission, False: Use unmodulated pixel transmission
    :param lr:              Learning Rate for ADAM optimizer
    :param n_iter:          Number of iterations used by ADAM optimizer
    :return:
    """

    #
    if illmod ==True:
        T_col = "T_ds"
    else:
        T_col = "Till_ds"

    wavs_ds,hs_ds,ill_ds = downsample_hs(hslocs,wavelengths,illumination,n_ds,n_k)
    unmix_endframe = np.identity(n_ds)
    spectra = []
    losses = []
    for h in hs_ds.index:
        hs = hs_ds.iloc[h]
        [x0, x1, y0, y1] = [hs.x0, hs.x1, hs.y0, hs.y1]
        P = image[y0:y1, x0:x1].reshape((y1 - y0) * (x1 - x0)).astype("float32")
        if grouped == True:
            labels = hs.Tk_group
            Tk, Pk = [], []
            for k in np.unique(labels):
                Tk.append(hs[T_col][:, labels == k].mean(axis=1))
                Pk.append(P[labels == k].mean())
            weights, lossarray = opti_unmix_pix_with_loss(endmembers=unmix_endframe, wavelengths=wavs_ds,
                                                          pixvals=np.array(Pk), transmission=np.array(Tk).T,
                                                          lr=lr, n_iter=n_iter)
            spectra.append(np.matmul(weights, unmix_endframe))
            losses.append(lossarray)
        else:
            Th = hs[T_col]
            weights, lossarray = opti_unmix_pix_with_loss(endmembers=unmix_endframe, wavelengths=wavs_ds,
                                                          pixvals=np.array(P), transmission=np.array(Th).T,
                                                          lr=lr, n_iter=n_iter)
            spectra.append(np.matmul(weights, unmix_endframe))
            losses.append(lossarray)
    spectra = np.array(spectra)
    return spectra, losses

# %% Region of Interest Spectral Plotting

# Spectra along a line

# %% TODO Standardised Presentation
import matplotlib.pyplot as plt


