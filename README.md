# machine_learning

## Connectome-Based Predictive Modelling
### Introduction

  Connectivity-based analysis in Neuroscience takes the connectivity matrix of the individual brain areas and uses it as an independent variable to predict specific characteristics. The connectivity matrix is obtained by correlating each pair of brain areas. Each entry of the connectivity matrix is called an “edge” and it depicts one brain region's influence on the other region. One specific model that uses the connectivity matrix to predict behavior is the Connectome-based predictive modelling (CBPM) (Finn et al., 2015; Finn & Bandettini, 2021; Shen et al., 2017), which selects the most important edges in the connectivity matrix based on significance and build a linear prediction model. Past studies that utilized this model have used individual connectivity matrices to predict behavior (Finn & Bandettini, 2021) and identify subjects from a large group (Finn et al., 2015).  For a more detailed explanation of the flow chart of the CBPM, please check out the presentation section.
  
![image of flow chart](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/CBPM%20flowchart.png)

*Flow chart of the Connectome Based Predictive Modelling*


### Past Analysis

  Most analyses used the Spearman correlation coefficient between the observed and predicted scores across subjects to assess and compare the prediction accuracy. The value of the correlation coefficient is expected to lie between 0.2 to 0.5. For instance, in Finn & Bandettini’s (2021) paper, using CBPM to predict cognition or behavior scores by resting state or movie-based functional connectivity, the correlation coefficient lies between 0.1 and 0.4 for different conditions. 

  Please refer to [Finn & Bandettini (2021)](https://www.sciencedirect.com/science/article/pii/S1053811921002408?via%3Dihub), [Shen et al. (2017)](https://www.nature.com/articles/nprot.2016.178), and  [Finn et al. (2015)](https://www.nature.com/articles/nn.4135) for more details of the application of CBPM. [Finn & Bandettini (2021)](https://www.sciencedirect.com/science/article/pii/S1053811921002408?via%3Dihub) used CBPM to predict behavior scores from restins state fMRI data as well as fMRI data when participants are watching movie. [Shen et al. (2017)](https://www.nature.com/articles/nprot.2016.178) introduce the CBPM in detail and compare it to other data modelling methods in neurosciences and states its pros and cons. [Finn et al. (2015)](https://www.nature.com/articles/nn.4135) used brain connectivity to identify individuals from a large group of participants. 

### Our Analysis

  We applied the CBPM method to the perspective fMRI data obtained by our past experiment (for a detailed description, please visit: https://github.com/fahsuanlin/labmanual/wiki/25.-Sample-data:-perspective-taking-fMRI-data). We utilized the same preprocessing method but calculated the connectivity matrix before our analysis by calculating the correlation coefficient between each pair of brain areas. 

### Results

  The correlation coefficient between the predicted and actual values typically lies between 0.2 and 0.5. Using Pearson correlation coefficient to relate connectivity matrix and behavioral measures then using thresholding to select edges that are significant (p<0.01), the correlation coefficient of the actual and predicted value of the model with the best performance is 0.2698 for the negative model and 0.5009 for the positive model for our data.
  
  In addition to using Pearson correlation to relate connectivity matrix and behavioral measures and using thresholding to select edges that are significant, we also tried out different variations of correlation methods including rank correlation and robust regression and feature selection methods including sigmoidal weighting. Our results showed that for the positive model, using robust regression and thresholding yields the best result (r = 0.508, **Figure 2**) while for the negative model using rank correlation and sigmoidal weighting yields the best result (r = -0.786, **Figure 3**). A more comprehensive result is shown in **Figure 1**.
  
  ![table of result](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/results%20table.png)
  
  ***Figure 1**: Correlation Coefficient between Predicted vs. Actual value*
  
  ![predicted vs. actual positive](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/positive%20model%20predicted%20vs.%20actual.png)
  
  
  ***Figure 2**: Predicted vs. Actual plot and the fitted line for positive model (Robust regression/Threshold)*
  
  ![predicted vs. actual negative](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/negative%20model%20predicted%20vs.%20actual.png)
  
  ***Figure 3**: Predicted vs. Actual plot and the fitted line for negative model (Rank Correlation/Sigmoidal weighting)*
  
  
&nbsp;
&nbsp;
&nbsp;
&nbsp;
## Machine Learning for Spatio-Temporal Brain Data (STBD)

### Introduction

The practical achievements of machine learning have been recognized in a wide range of scientific research fields, and its application in brain science and medical imaging is bound to be the mainstream trend. Due to the introduction of the concept of deep learning in 2006 (Geoffrey et al.) and the successful application of deep learning models such as Convolutional Neural Network (CNN) since 2012, in general, we can divide machine learning algorithms into conventional machine learning and deep learning. In the field of brain imaging research, conventional machine learning algorithms that have been successfully applied generally include: Support Vector Machine (SVM), Ensemble, Logistic Regression (LR), K Nearest Neighbours (KNN) etc. ; and deep learning algorithms generally include: CNN, Deep Belief Network (DBN), Stacked Autoencoder (SAE), etc. and related derivative algorithms. (Mamoon et al. 2020) Although the practice in recent years shows that deep learning seems to be more mainstream, in fact, when faced with different problems and goals, the two have their own advantages. The difference mainly exists in: the automation of feature engineering, computing power requirements, computational efficiency, algorithm interpretability, etc.

At the same time, the data input format is also an important reference when we choose the algorithm. Spatio-Temporal Brain Data (STBD) is the main content of data analysis in our lab, that is, the data contains both temporal and spatial features, mainly including EEG and fMRI. This is just like in the video recognition task in the field of computer vision, which is difficult to describe with only single-frame image information but would have enough potential to realize the target by utilizing the whole video composed of image sequence with a fixed frame rate. For EEG data, the spatial feature can be the corresponding positional relationship of the acquisition channels, while the temporal feature can be the EEG signal changes of each channel; for fMRI data, the spatial feature can be the 3D voxel distribution, while the temporal feature is the BOLD signal changed in each voxel.

Therefore, we have reason to believe that, when machine learning is applied to STBD, we can obtain information from two levels of space and time, respectively, and form connections with each other. Taking fMRI data as an example, traditional research methods generally apply models to two levels separately, such as using General Linear Model (GLM), Independent Component Analysis (ICA), Sparse Dictionary Learning (SDL) and recently deep learning models (such as CNN for spatial information and RNN for temporal sequence). In recent years, more end-to-end models that combine the two aspects have been proposed, such as Spatio-Temporal Convolutional Neural Network (ST-CNN) that combines 3D U-Net and 1D CAE (You et al., 2020), Spatio-Temporal Graph Convolutional Network (ST-GCN) combines GCN and TCN (Soham et al., 2020) etc. At the same time, conventional machine learning may still play a better role when fMRI, EEG, or behavioral data have been artificially processed to obtain relevant feature representations. It can be considered that the practice of the above algorithms and models can inspire us to apply machine learning in STBD.

In addition, the goal of the problem is also an important reason for us to choose different models. For example, the difference between rs-fMRI and tfMRI, or between classification and regression problems, etc. Regarding the application areas of different problem goals, common areas include: localizing the activated brain region, detecting the brain information pathways, diagnosing disease, judging psychological states, etc. (Nikola et al., 2017)

### Our Analysis

One of our studies focuses on using already extracted features (including fMRI, EEG, behaviors, etc.), rather than raw data, to predict subjects’ corresponding scores on standardized tests of relevant domains. At the same time, the data itself also has the following characteristics that may interfere with the model: 1) The number of samples is relatively small; 2) Some data are missing in the collection of original data; 3) The data types and ranges of different features are not uniform.

For such a dataset, we first perform reasonable preprocessing on the dataset (including data cleaning, normalization, dimensionality reduction, etc.), and then apply conventional machine learning regression models (including SVR, MLP, LASSO, ELASTIC, etc.) to make corresponding predictions. Finally, we use the Root Mean Squared Error (RMSE) under five-fold cross-validation to evaluate the accuracy of the model prediction, and the results for different targets are shown in Figure 1.

![RMSE of 9 lables](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Kaihua_RMSE_of_9_lables.png)

***Figure 1**: RMSE of the prediction results of different regression models on the training set and testing set* 

At the same time, we can reflect the model’s accuracy more intuitively by drawing the distribution figure with the actual values as horizontal axis and the predicted value as the vertical axis. Take one training process of LANGUAGE_ST_RAW_SIMILARITY, one of the targets, as an example, the results are shown in Figure 2.

![scatter plot predicted vs actual](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Kaihua_scatter_plot_predicted_vs_actual.png)

***Figure 2**: Distribution of the actual value versus the predicted value* 

### Future work

Although research groups in related fields including our laboratory are making steady progress, there are still many tasks in this field that are difficult to obtain good results, which are also what we are trying to solve. Regarding the prospect of future work, maybe it can be improved in the following ways.

* Due to the relatively complex collection of data such as fMRI, the number of relevant samples is small, which often leads to overfitting of complex models, making it difficult to obtain good results. Therefore, more collection of high-quality data can be considered.
* Relevant data may exist noise due to the behavior of subjects and the surrounding environment of the device. Therefore, designing a better noise reduction algorithm to remove the relevant noise and better achieving further feature dimensionality reduction can be considered.
* Models for classification/regression can learn from the machine learning algorithms in other fields, such as the Transformer models that have developed rapidly in recent years, which may have potential in our field.
* ...


&nbsp;
&nbsp;

## Data and Background

fMRI scans use the low frequency Blood Oxygen Level Dependent signals as a measure of regional brain activity. For the purposes of data analysis and modeling, the information can be thought of as a time series of 3D matrices. Each time step would represent a functional scan of the subject's brain and the number of time steps per experiment will differ according to the requirements. The dimensions of each scan can be represented by the tuple ( T, X, Y, Z ) while T in the temporal dimension ( number of time steps) and X, Y, Z are the spatial dimensions. A single value given by a specific temporal and spatial location ( eg: t,x,y,z) is referred to as a Voxel (a volumetric pixel).

![Single Time Step Dimensions](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan_1.PNG)

![Single Time Step Dimensions](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan_2.PNG)

(Single Time Step Dimensions)
### Basic Data Loading

fMRI data collection and experimental design are extremely time consuming tasks. Hence it is often the case that you would be working with a very limited number of samples in your dataset. This low sample size along with the incredibly large number of features collected per scan are the major obstacles to modeling and generalising the models for fMRI based prediction. Fortunately, there have been efforts by the neuroimaging community to collect sizable datasets through coordinated efforts. The [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (HCP) is one such initiative, which deals with both resting state and task based experiments with around 1000 samples per experiment. This data has been made available to the public under various packaged releases.

You may download processed or unprocessed data from the link given above. A sample script to load the unprocessed data and plot it in a slice-wise fashion can be found [here](https://colab.research.google.com/drive/1Ecjr8nQs0FD8GGv93CTKw2S5t0elf9_B?authuser=2#scrollTo=fDtL_oN1qFqw).

![Single 2-D Slice Plot](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan_3.PNG)
(Single 2-D Slice Plot)

Other than using the raw scan data we can focus our attention on the regions of interest on the cortical surface, substantially reducing the dimensionality of the data we need to deal with. Common practice and empirical evidence based analysis has led to the parcellation of the brain in 75 ROIs (regions of interest) per hemisphere. This given us 10242 voxels of information per hemisphere.

![Cortical Surface with ROIs](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/Ramanathan_4.PNG)
(Cortical Surface with ROIs)

## Representation Learning
Even with just the cortical surface voxels, the dimensionality of the data is still quite large. A useful method to reduce dimensionality while retaining the maximum amount of information is learning a compact representation of the data. A common approach used to achieve this is an Autoencoder (AE) model. The idea is to learn a nonlinear function: f(x) : y called the encoder and another similar function: g(y) : x’ called the decoder. The goal being for x’ to approximate x as accurately as possible. A VAE (variational autoencoder) is a modified version of a simple AE which attempts to implicitly cluster the data in the embedded representation (y).

Insert Image
(VAE Architecture)

This idea can be used to reduce the dimensionality of the fMRI features. There are 2 ideas that can be explored. The use of 3D convolutions on full volumetric data i.e. a convolutional autoencoder. [Itzik Malkiel et al](https://arxiv.org/pdf/2112.05761.pdf) suggests a pertaining via representation learning using the HCP dataset. While this idea has been used in other domains, the size of each fMRI scan sample makes it a bit impractical compared to traditional image AEs for training. We plan to use the pertaining idea for spatial and temporal encoding while modifying the models used to achieve those results.

Combining the ideas of using the cortical voxels and a convolutional AE, taking inspiration from the [spherical U-net] (https://github.com/zhaofenqiang/Spherical_U-Net) paper, we can keep the size of the data within acceptable limits while retaining most of the information. Following this we will utilise and test various temporal information extraction techniques and test the validity of the pretrained models on our perspective dataset.

&nbsp;
&nbsp;
&nbsp;
&nbsp;
