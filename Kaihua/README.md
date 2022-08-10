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
&nbsp;
&nbsp;
