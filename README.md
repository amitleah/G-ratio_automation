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

## Purpose of the code:

In the community of labs studying the nervous system, the quantification of a parameter called “g-ratio” is critical. This parameter takes a cross section image of a nerve (obtained by Electron Microscopy) and measures the diameter of single fibers (axons) with and without the insulating layer (called Myelin). This ratio indicates the velocity in which electrical signals travel in the fibers and intern reflect on the properties and health of the nerve. 

Currently, g-ratio quantification is done manually in my lab, a task which is tedious and extremely time-consuming. Using Open Computer Vision Library (OpenCV), this python code is able to detect the outline of the myelin layer and calculate the g-ratio automatically after setting a single threshold parameter for each image. This allows to collect data on a much larger scale in just a few minutes. 

## How to use the code:

Once code is running, a threshold bar and 4 stacked images with/out filters will show up as follows:

![alt text](/github_images/diagram1.PNG)

click "C" to see the countours created. Play with threshold bar until satisfied.
Then click "S" for a few seconds in order to export this data to Excel, and move on to the next image. 

![alt text](/github_images/diagram2.PNG)

The excel table exported includes details on every insulated fiber that was detected. 
Each row is a different fiber, consisting of the inner and outer outline.
Columns include the following details for each fiber: 
Inner and outer circle area, Inner and outer circle radius, axon diameter, genes (group it belongs to - WT or KO) and finally g-ratio. 


![alt text](/github_images/diagram3.PNG)


After going through all the images, a scatter plot will pop up: comparing 2 groups: “WT” (untreated) in red, and “KO” (treated) in blue. This is the classic plot used in papers by scientists studying electrical current in nerve cells, such as my lab. 
