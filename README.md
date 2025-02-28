# Deep-ER: Deep Learning ECCENTRIC Reconstruction for fast high-resolution neurometabolic imaging

## Description
The Deep-ER neural network was specially developed to reconstruct non-cartesian compressed sense MRSI. We demonstrated that Deep-ER provides high efficiency and quality: 
1. The reconstruction time of whole-brain high-resolution 1H-MRSI 3D FID-ECCENTRIC is greatly reduced by a factor of almost 600 using Deep-ER compared to conventional TGV-ER.
2. Efficient usage of GPU memory enables multichannel high-resolution MRSI data processing.
3. High temporal consistency across accelerations reduces spectral noise and improves precision and accuracy of metabolite quantification. 
4. Sharper spatial features and less image blurring are achieved with increasing accelerations. Although we demonstrated Deep-ER for undersampled ECCENTRIC trajectories, the network can be easily applied to other accelerated spectral–spatial encoding trajectories.

### Architecture
In this work, we built on and extended the fully convolutional joint-domain Interlacer[^1] architecture for effective domain transfer to MRSI reconstruction. The convolutional network takes as input to the first layer the coil-combined undersampled image data ($x_1$) and gridded multi-channel undersampled k-space ($k_1$) to predict fully sampled k-space on a Cartesian grid, which is subsequently transformed to image space.

Each Interlacer layer separately applies convolutional blocks in k-space and image-space. Before each block, image- and k-space features are merged via weighted addition with learnable mixing parameters $\{\alpha_i, \beta_i\}$,
```math
k_i^{mix} = \alpha_i \mathcal{F}(\mathcal{C}^{-1}(x_i)) + (1-\alpha_i) k_i
```
```math
x_i^{mix} = \beta_i x_i + (1-\beta_i) \mathcal{F}^{-1}(\mathcal{C}(k_i))
```
where $\mathcal{F}$ represents the FFT operation and $\mathcal{C}$ the channel-wise coil combination of individual coil images by voxel-wise multiplication with ESPIRiT profiles[^2]. 

<p align="center" width="100%">
  <img width="50%" src="https://github.com/user-attachments/assets/798ac957-1868-4ab6-b6fc-282f57798179">
</p>

### Deep learning MRSI reconstruction
The ground truth image $x_{GT}$ was generated for training purposes from fully sampled k-space data $k_{FS}$ , utilizing the conventional reconstruction method presented in Klauser et al.[^3], which is based on an iterative optimization that employs Total-Generalized-Variation (TGV) [^4] as a regularizer.

### Results
In-vivo metabolic images reconstructed from two-fold accelerated (A.F.=2) ECCENTRIC data acquired in a glioma patient and a healthy volunteer are shown in the figure below. Metabolic maps in the patient show well defined boundaries for the tumor and metabolic heterogeneity within the tumor. There is higher contrast between the tumor and the normal brain in the maps produced by Deep-ER compared to TGV. In the healthy volunteer similar gray-white matter structural features are visible in the metabolic maps obtained by both Deep-ER and TGV reconstructions. 

<p align="center" width="100%">
    <img width="90%" src="https://github.com/user-attachments/assets/724e391e-5f1b-4ec6-aff0-1cec6bf0f25e">
</p>


## Getting Started
This repository provides the code to train a neural network in 'src', including a 'run.py' and 'config.py' file. The 'run.py' file is intended for execution, the corresponding parameters can be changed in the 'config.py' file.

### Training Data
Due to privacy regulations and institutional policies, these data cannot be made publicly available in a global repository. However, access to the data may be granted upon reasonable request and subject to a data transfer agreement (DTA) with the respective institutions.

### Train your own Deep-ER
To start the training of WALINET with the previously generated data, the 'run.py' file has to be executed.
```
python run.py
```
Please modify the 'config.py' file to adjust the number of epochs and other training parameters.


### References
[^1]: Singh, Nalini M., Iglesias, Juan Eugenio, Adalsteinsson, Elfar, Dalca, Adrian V., Golland, Polina, 2022. Joint frequency and image space learning for MRI reconstruction and analysis. J. Mach. Learn. Biomed. Imaging 2022
[^2]: Uecker, Martin, Lai, Peng, Murphy, Mark J., Virtue, Patrick, Elad, Michael, Pauly, John M., Vasanawala, Shreyas S., Lustig, Michael, 2014. ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. Magn. Reson. Med. 71 (3), 990–1001.
[^3]: Klauser, Antoine, Strasser, Bernhard, Bogner, Wolfgang, Hingerl, Lukas, Courvoisier, Sebastien, Schirda, Claudiu, Rosen, Bruce R, Lazeyras, Francois, Andronesi, Ovidiu C, 2024. ECCENTRIC: a fast and unrestrained approach for high-resolution in vivo metabolic imaging at ultra-high field MR. Imaging Neuroscience 2, 1–20.
[^4]: Knoll, Florian, Bredies, Kristian, Pock, Thomas, Stollberger, Rudolf, 2011. Second order total generalized variation (TGV) for MRI. Magn. Reson. Med. 65 (2), 480–491.
