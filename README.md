# Python Bootcamp Final Project - G ratio automation

###### Diagram of nerve cell
![alt text](/github_images/diagram0.PNG)

## Description
Automation of g-ratio measurement

## Installation
Use the package manager pip to install G-ratio_automation.

```bash
pip install -r requirements.txt
```
You can download the Images 'test' folder from Google Drive. Name this folder "Images".
https://drive.google.com/drive/folders/1CdBMGzpU9LzXIdJjpZ-jwynI9IWbyea4?usp=sharing


## Purpose of the code:

In the community of labs studying the nervous system, the quantification of a parameter called “g-ratio” is critical. This parameter takes a cross section image of a nerve (obtained by Electron microscopy) and measures the diameter of single fibers (axons) with and without the insulating layer (called myelin). This ratio indicates the velocity in which electrical signals travel in the fibers and intern reflect on the properties and health of the nerve. 

Currently, g-ratio quantification is done manually in my lab, a task which is tedious and extremely time-consuming. Using Open Computer Vision Library (OpenCV), this python code is able to detect the outline of the myelin layer and calculate the g-ratio automatically after setting a single threshold parameter for each image. This allows to collect data on a much larger scale and accuracy in just a few minutes. 

## How to use the code:

Once code is running – a threshold bar and 4 stacked images with/out filters will show up as follows:

![alt text](/github_images/diagram1.PNG)

click "C" to see the countours created. Play with threshold bar until satisfied.
then click "S" for a few seconds in order to export this data to Excel. 

![alt text](/github_images/diagram2.PNG)

The excel table exported includes details on every insulated fiber that was detected. 
Each row is a different fiber, consisting of the inner and outer outline.
Columns include the following details for each fiber: 
Inner and outer circle area, Inner and outer circle radius, axon diameter, genes (group it belongs to) and finally g-ratio. 


![alt text](/github_images/diagram3.PNG)


After going through al the images, This plot will pop up: comparing 2 groups: “WT” (untreated) in red, and “KO” (treated) in blue. This is the classic plot used in papers by scientists studying the nervous system and electrical signal conductance, such as my lab. 
