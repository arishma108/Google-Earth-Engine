# Google-Earth-Engine

## Implementation of the Thornthwaite-Mather procedure to map groundwater recharge

### Groundwater Recharge Estimation using the Thornthwaite-Mather Procedure

#### Welcome to my GitHub repository! 
This project focuses on estimating groundwater recharge, which represents the amount of water from precipitation that reaches the groundwater table. By determining groundwater recharge, we can gain a better understanding of the available and renewable groundwater in watersheds, as well as the shape of groundwater flow systems.

#### Introduction
One of the simplest methods to estimate groundwater recharge is the Thornthwaite-Mather procedure, as published by Thornthwaite and Mather in 1955 and 1957. This procedure calculates the water balance in the root zone of the soil, considering factors such as evaporation, transpiration, soil storage, and infiltration.

#### Procedure Overview
The Thornthwaite-Mather procedure relies on the following parameters and variables:
##### Soil texture information: 
This includes the sand and clay content, which describe the hydraulic properties of the soil and its capacity to store and infiltrate water.
##### Meteorological records: 
Precipitation and potential evapotranspiration data are required for the estimation.

Please note that this procedure does not take into account other factors that can influence groundwater recharge, such as terrain slope, snow cover, crop/land cover variability, and irrigation.

#### Overview
This is divided into four parts:
##### Initialization and Setup: 
In this part, we will initialize the Earth Engine Python API, import necessary libraries, and define the location and period of interest.
##### Exploring Soil Properties: 
Here, we will explore OpenLandMap datasets related to soil properties. We will calculate the wilting point and field capacity of the soil using mathematical expressions applied to multiple images.
##### Importing Meteorological Data: 
This part focuses on importing evapotranspiration and precipitation datasets. We will define a function to resample the time resolution of an ee.ImageCollection and homogenize the time index of both datasets. Finally, we will combine the datasets into one.
##### Implementing the Thornthwaite-Mather Procedure: 
In the final part, we will implement the Thornthwaite-Mather procedure by iterating over the meteorological ee.ImageCollection. We will also compare groundwater recharge in two different locations and display the resulting mean annual groundwater recharge over France.

Feel free to explore the code and datasets provided in this repository. If you have any questions or suggestions, please don't hesitate to reach out. 
