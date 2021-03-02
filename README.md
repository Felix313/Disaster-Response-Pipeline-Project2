# Disaster Response Pipeline Project

## Table of Contents
[Installation](https://github.com/spreuhs/data_science_nd_project_3/blob/main/README.md#installation)

[Running Instructions](https://github.com/spreuhs/data_science_nd_project_3/blob/main/README.md#running-instructions)

[Project Summary](https://github.com/spreuhs/data_science_nd_project_3/blob/main/README.md#project-summary)

[Repository Structure](https://github.com/spreuhs/data_science_nd_project_3/blob/main/README.md#repository-structure)

[Acknowledgements](https://github.com/spreuhs/data_science_nd_project_3/blob/main/README.md#acknowledgements)

## Installation

The code should run without issues using Python versions 3.*.
Libraries neccessary to run the code:
	-pandas
	-sqlalchem
	-numpy
	-re
	-nltk
	-sklearn
	-pickle


## Running Instructions
- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- To run the web app.
    `python apps/run.py`

You can then enter the flask app on [http://0.0.0.0:3001/](http://0.0.0:3001/)

## Project Summary

Social Media is one important source for information about potential disasters. The huge amount of users leads to fast and detailed information as people tend to heavily use Social Media in case of major events. Unfortunately the result is a incredibly large number of messages which is hard to interpret and is impossible to scan by hand.

Therefore this projects categorizes given Twitter messages using machine learning in order to potentially support disaster responses in the future.


## Repository Structure

- data/ contains everything required for data preperation, including the data as .csv, a python script for the preperation and the final database
- models/ contains the the python script for model creation and the final classifier
- app/ contains files for the web data creation


## Acknowledgements

Thanks to Figure Eight for their data on Twitter messages.
 
