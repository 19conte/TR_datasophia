# TR DataSophia - Alexandre CONTE

## Presentation

This repo contains the data, codes and results of my Trimestre Recherche in DataSophia.  

Abstract :
*Recent studies on solar power forecasting have been using neural networks (NN) with sky-images. From these images taken by hemispheric cameras, the role of NN is to directly predict a value of either solar irradiance or PV production. Even though NN perform well, they lack interpretability, that is to say, transparency and robustness. They act as a black box since the way the prediction is made is hidden. In particular, it is hidden to field experts, who could be wary of the prediction. The ambition of this paper is to suggest a strategy to create an interpretable model, with an example in the field of solar forecasting.
From domain-specific interpretability definition to fine-tuning of the model, each step is meant to stick with domain knowledge.*  

In practice, we build a Generalized Additive Model using features extracted from sky images. For each image, the model's target is the K_c value, with :  

![](https://latex.codecogs.com/gif.latex?K_c&space;=&space;\frac{GHI}{GHI_{clearsky}})

## Data provided by Solais

Thanks to the company Solais, I had access to sky images from an hemisperic camera, as well as pyranometers GHI measurements. Sky images taken from June to October 2020 are in the **Solais_Data/FTP folder/**, and GHI time series are stored in csv files.  

A clean K_c dataset is built step by step in a jupyter notebook **src/OneTask_nbooks/ghi_dataset.ipynb**. Then it was needed to identify images files for which a K_c value is available in the dataset. This is done thanks to **src/OneTask_nbooks/extract_valid_paths.ipynb**.

## From which features are extracted

All the code of the project is in **src/**. Work on images and feature extraction techniques is done thanks to the scripts in **src/feature_extraction/**.

## And these data created from raw data is stored

In **preprocessed_data/** one can find images for the CNN (**preprocessed_data/mobotix1_prepro_240/**) and the CNN target values.  
The csv file **preprocessed_data/final_dataset.csv** contains all features extracted from images, and solar radiation data.

## Finally models are trained

Two models are trained : CNN predicting GHI (**src/models/CNN**) and a GAM predicting K_c. For the GAM, the first step was to find a ranking of feature importances, and then to build the model. This two steps are done in **src/models/feature_selection_and_GAM.ipynb**.

## The results

A few documents illustrating the models performances and a detailed article are available in **results/**.



