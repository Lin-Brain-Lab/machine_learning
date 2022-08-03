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
  
  * **Figure 1**: Correlation Coefficient between Predicted vs. Actual value*
  
  ![predicted vs. actual positive](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/positive%20model%20predicted%20vs.%20actual.png)
  
  
  * **Figure 2**: Predicted vs. Actual plot and the fitted line for positive model (Robust regression/Threshold)*
  
  ![predicted vs. actual negative](https://github.com/Lin-Brain-Lab/machine_learning/blob/main/negative%20model%20predicted%20vs.%20actual.png)
  
  * **Figure 3**: Predicted vs. Actual plot and the fitted line for negative model (Rank Correlation/Sigmoidal weighting)*
  
  
