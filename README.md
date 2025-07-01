# Head and Neck Cancer Patient Survival Prediction

## Course Information
- **Course**: ECE-UY 4563 Introduction to Machine Learning
- **Institution**: New York University
- **Semester**: Fall 2024
- **Instructor**: Prof. Sundeep Rangan

## Project Overview
This project focuses on predicting cancer (specifically head and neck cancer) patient outcomes through binary survival classification using various machine learning approaches. We implemented and compared the performance of three different models:
- Logistic Regression
- Support Vector Machine (SVM)
- Neural Network

## Team Members
- Hongtai Du (hd2609@nyu.edu)
- Daniel Yang (ty2184@nyu.edu)

## Data Source
The dataset used in this project was obtained from The Surveillance, Epidemiology, and End Results (SEER) Program. Due to data privacy and usage agreements, the sample dataset cannot be uploaded on GitHub. Here are the insgtructions we get data samples:
1. Search for SEER database in browser to register for an account and sumbit request access to SEER Data
2. Check email for the download link, and install the software
3. Click on FIle -> New -> Case Listing Session
4. In "Data", select "Incidence - SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)". Then move to "Selection".
5. Choose following variables: Age recode with <1 year olds, Sex, Year of diagnosis, Race recode (W, B, AI, API), Primary Site, Histologic Type ICD-O-3, Combined Summary Stage (2004+), Surgery of oth reg/dis sites (1998-2002), Chemotherapy recode (yes, no/unk), Radiation recode, Vital status recode (study cutoff used), Survival months, SEER cause-specific death classification, Tumor Size Summary (2016+), EOD Primary Tumor (2018+), CS lymph nodes (2004-2015), CS tumor size (2004-2015)
6. Execute file
7. Export result matrix to csv or txt file

## Models and Analysis
We conducted a comparative analysis of three machine learning models:
1. Logistic Regression - A baseline model for binary classification
2. Support Vector Machine - For potentially better handling of non-linear relationships
3. Neural Network - To capture complex patterns in the data

The performance metrics and detailed analysis of each model can be found in the project documentation.

---
*Note: This project is part of the academic curriculum at NYU Tandon School of Engineering.*
