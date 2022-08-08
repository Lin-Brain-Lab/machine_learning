# Data and Background

fMRI scans use the low frequency Blood Oxygen Level Dependent signals as a measure of regional brain activity. For the purposes of data analysis and modeling, the information can be thought of as a time series of 3D matrices. Each time step would represent a functional scan of the subject's brain and the number of time steps per experiment will differ according to the requirements. The dimensions of each scan can be represented by the tuple ( T, X, Y, Z ) while T in the temporal dimension ( number of time steps) and X, Y, Z are the spatial dimensions. A single value given by a specific temporal and spatial location ( eg: t,x,y,z) is referred to as a Voxel (a volumetric pixel).

![Single Time Step Dimensions](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan/Ramanathan_1.PNG)

![Single Time Step Dimensions](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan/Ramanathan_2.PNG)

(Single Time Step Dimensions)
# Basic Data Loading

fMRI data collection and experimental design are extremely time consuming tasks. Hence it is often the case that you would be working with a very limited number of samples in your dataset. This low sample size along with the incredibly large number of features collected per scan are the major obstacles to modeling and generalising the models for fMRI based prediction. Fortunately, there have been efforts by the neuroimaging community to collect sizable datasets through coordinated efforts. The [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (HCP) is one such initiative, which deals with both resting state and task based experiments with around 1000 samples per experiment. This data has been made available to the public under various packaged releases.

You may download processed or unprocessed data from the link given above. A sample script to load the unprocessed data and plot it in a slice-wise fashion can be found [here](https://colab.research.google.com/drive/1Ecjr8nQs0FD8GGv93CTKw2S5t0elf9_B?authuser=2#scrollTo=fDtL_oN1qFqw).

![Single 2-D Slice Plot](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan/Ramanathan_3.PNG)
(Single 2-D Slice Plot)

Other than using the raw scan data we can focus our attention on the regions of interest on the cortical surface, substantially reducing the dimensionality of the data we need to deal with. Common practice and empirical evidence based analysis has led to the parcellation of the brain in 75 ROIs (regions of interest) per hemisphere. This given us 10242 voxels of information per hemisphere.

![Cortical Surface with ROIs](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan/Ramanathan_4.PNG)
(Cortical Surface with ROIs)

## Representation Learning
Even with just the cortical surface voxels, the dimensionality of the data is still quite large. A useful method to reduce dimensionality while retaining the maximum amount of information is learning a compact representation of the data. A common approach used to achieve this is an Autoencoder (AE) model. The idea is to learn a nonlinear function: f(x) : y called the encoder and another similar function: g(y) : x’ called the decoder. The goal being for x’ to approximate x as accurately as possible. A VAE (variational autoencoder) is a modified version of a simple AE which attempts to implicitly cluster the data in the embedded representation (y).

Insert Image
(VAE Architecture)

This idea can be used to reduce the dimensionality of the fMRI features. There are 2 ideas that can be explored. The use of 3D convolutions on full volumetric data i.e. a convolutional autoencoder. [Itzik Malkiel et al](https://arxiv.org/pdf/2112.05761.pdf) suggests a pertaining via representation learning using the HCP dataset. While this idea has been used in other domains, the size of each fMRI scan sample makes it a bit impractical compared to traditional image AEs for training. We plan to use the pertaining idea for spatial and temporal encoding while modifying the models used to achieve those results.

Combining the ideas of using the cortical voxels and a convolutional AE, taking inspiration from the [spherical U-net](https://github.com/zhaofenqiang/Spherical_U-Net) paper, we can keep the size of the data within acceptable limits while retaining most of the information. Following this we will utilise and test various temporal information extraction techniques and test the validity of the pretrained models on our perspective dataset.
