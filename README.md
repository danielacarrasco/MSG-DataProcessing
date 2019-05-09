Overview 
----------

Scripts to process wearables raw data and produce EDF files for upload to the Seer platform. 


Empatica E4
----------

The script empatica_edf.py processes the data from Empatica E4 devices. 

The code uses the following raw files from E4 manager:
- ACC.csv
- BVP.csv
- EDA.csv
- HR.csv
- TEMP.csv

To run, update the following parameters in the code:
- loadPath: Path to the files for processing.
- savePath: Patch where the folder (edf/) with the EDF files will be saved. 
- time_offset: Time offset between the video/EEG and the wearable device. Positive or negative.

Run empatica_edf.py

GENEActiv
----------

The script geneactiv_edf.py processes the data from GENEActiv devices.

The code uses the csv file from GENEActiv PCSoftware (converted from .bin to .csv):
- Output.csv 

To run, update the following parameters in the code:
- loadPath: Path to the file for processing, including the name of the .csv file.
- savePath: Path where the folder (edf/) with the EDF files will be saved. 
- time_offset: Time offset between the video/EEG and the wearable device. Positive or negative.
- device_n: Device seruial number.

Run geneactiv_edf.py.

Requirements
----------

- pyedflib https://pyedflib.readthedocs.io/en/latest/#
