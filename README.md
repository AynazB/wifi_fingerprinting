# Wi-Fi Fingerprinting and Location Analytics (R Markdown)

## Dataset

The UJIIndoorLoc database covers three buildings of Universitat Jaume I with 4 or more floors and almost 110.000m2. It can be used for classification, e.g. actual building and floor identification, or regression, e.g. actual longitude and latitude estimation. It was created in 2013 by means of more than 20 different users and 25 Android devices. The database consists of 19937 training/reference records (trainingData.csv file) and 1111 validation/test records (validationData.csv file).

_Source: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc)._

## Goal

The main goal of this task is to evaluate the application of machine learning techniques to the problem of indoor localization via Wi-Fi fingerprinting.

## Attributes

Attribute 001 (WAP001) to 520 (WAP520): Intensity value for the WAP. Negative integer values from -104 to 0 and +100. Positive value 100 used if the WAP was not detected.  
Attribute 521 (Longitude): Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000.  
Attribute 522 (Latitude): Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018.  
Attribute 523 (Floor): Altitude in floors inside the building. Integer values from 0 to 4.  
Attribute 524 (BuildingID): ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2.  
Attribute 525 (SpaceID): Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values.  
Attribute 526 (RelativePosition): Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values.  
Attribute 527 (UserID): User identifier (see below). Categorical integer values.  
Attribute 528 (PhoneID): Android device identifier (see below). Categorical integer values.  
Attribute 529 (Timestamp): UNIX Time when the capture was taken. Integer value.  
