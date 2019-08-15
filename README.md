# USAL Echocardiogram Analysis

This project classifies echocardiographic DICOM images.


## Table of contents

1. [Introduction](https://github.com/dssg/usal_echo#introduction)
2. [Overview](https://github.com/dssg/usal_echo#overview)
3. [Infrastructure requirements](https://github.com/dssg/usal_echo#infrastructure-requirements)
4. [Installation and setup](https://github.com/dssg/usal_echo#installation-and-setup)
5. [Pipeline options](https://github.com/dssg/usal_echo#pipeline-options)
6. [Code organisation](https://github.com/dssg/usal_echo#code-organisation)
7. [Contributors](https://github.com/dssg/usal_echo#contributors)
8. [License](https://github.com/dssg/usal_echo#license)

## Introduction

### Data Science for Social Good at Imperial College London 2019

The Data Science for Social Good Fellowship is a summer program to train aspiring data scientists to work on data mining, machine learning, big data, and data science projects with social impact. Working closely with governments and nonprofits, fellows take on real-world problems in education, health, energy, public safety, transportation, economic development, international development, and more.

For three months they learn, hone, and apply their data science, analytical, and coding skills, collaborate in a fast-paced atmosphere, and learn from mentors coming from industry and academia.

### Partners

The project was done in collaboration with the CIBERCV (Biomedical Research Networking Centres - Cardiovascular) research team working at the Hospital Universitario de Salamanca (USAL). USAL has one of the most advanced cardiographic imaging units in Spain and serves an ageing, largely rural population. The team of cardiologists at USAL is investigating new technologies such as artificial intelligence to help improve patient care.

## Overview

The processing pipeline is structured as follows.
![USAL Echo Project Overview](docs/images/usal_echo_pipeline_overview.png?raw=true "USAL Echo Project Overview")

## Infrastructure requirements
We retrieve our data from an AWS S3 bucket and use an AWS EC2 server for running all code. Results for each processing layer are stored in an AWS RDS.
```
Infrastructure: AWS

+ AMI: ami-079bef5a6246ca216, Deep Learning AMI (Ubuntu) Version 23.1
+ EC2 instance: p3.2xlarge
    + GPU: 1
    + vCPU: 8
    + RAM: 61 GB
+ OS: ubuntu 18.04 LTS
+ Volumes: 1
    + Type: gp2
    + Size: 450 GB
+ RDS: PostgreSQL
    + Engine: PostgreSQL
    + Engine version: 10.6
    + Instance: db.t2.xlarge
    + vCPU: 2
    + RAM: 4 GB
    + Storage: 40 GB
```

## Installation and setup

#### 0. Requirements
In addition to the infrastructure mentioned above, the following software is required:
* [Anaconda](https://docs.anaconda.com/anaconda/install/)
* [git](https://www.atlassian.com/git/tutorials/install-git)  

The instructions below assume that you are working setting up the repository in your terminal.

#### 1. Conda env and pip install
Clone the TensorFlow Python3 conda environment in your GPU instance set up with AWS Deep Learning AMI and activate it. Then install the required packages with pip.
```
conda create --name usal_echo --clone tensorflow_p36
conda activate usal_echo
pip install -r requirements.txt
```

#### 2. Clone repository
After activating your Anaconda environment, clone this repository into your work space:  
```
git clone https://github.com/dssg/usal_echo.git
```

Navigate into your newly cloned `usal_echo` diretctory and run the setup.py script.  
```
python src/setup.py
```

#### 3. Credentials files

To run the pipeline, you need to specify the credentials for your aws and postgres infrastructure. The pipeline looks for credentials files in specific locations. You should create these now if they do not already exist.

##### aws credentials   
Located in `~/user/.aws/credentials` and formatted as:
```
[default]
aws_access_key_id=your_key_id
aws_secret_access_key=your_secret_key
```
The pipeline uses the `default` user credentials.

##### postgres credentials  
Located in `~/user/.psql_credentials.json` and formatted as:
```
{
"user":"your_user",
"host": "your_server.rds.amazonaws.com",
"database": "your_database_name",
"psswd": "your_database_password"
}
```

#### 4. Run the pipeline
The final step is to run the `inquire.py` script. 
```
python src/inquire.py
```
This will launch a questionnaire in your command line that takes you through the setup options for running the pipeline. The options are discussed in detail in the following section.

## Pipeline options

Show a new user how to use the package.

## Code organisation

The code is organised as follows:
1. `d00_utils`: Utility functions used throughout the system
2. `d01_data`: Ingesting dicom metadata and XCelera csv files from s3 into database
3. `d02_intermediate`: Cleaning and filtering database tables, downloading, decompressing and extracting images from dicom files for experiments
4. `d03_classification`: Classification of dicom images in image directory
5. `d04_segmentation`: Segmentation of heart chambers
6. `d05_measurements`: Calculation of measurements from segmentations
7. `d06_reporting`: Results analysis against machine type and BMI 
8. `d07_visualisation`: Generating plots for reporting

## Contributors

**Research fellows**: Courtney Irwin, Dave Van Veen, Wiebke Toussaint, Yoni Nachmany  
**Technical mentor**: Liliana Millán (Technical Mentor)  
**Project manager**: Sara Guerreiro de Sousa (Project Manager)  

## License

This codebase is made available under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

<p align="center">
<img src="docs/images/cardiologist_journey_3.jpeg" alt="Cardiologist at work" width="400" align:center/>
</p>
