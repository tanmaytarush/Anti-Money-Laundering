import pandas as pd
import numpy as np
import json
import csv
import logging
from random import randint

data_path = "AML_Data.csv"

logging.basicConfig(filename='logs/model_development.txt',
					filemode='a',
					format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("DATA PREPROCESSING 1 STAGE")

logging.warning("Reading Dataset...")

X = pd.read_csv(data_path)
X = X.to_numpy()

logging.warning("Read Dataset")

nameOrigCol = 3
nameDestCol = 6
nameOrig = []
nameDest = []
nameCount = {}
namesWithMoreThanOneOccurrence = []

logging.warning("Checking Each Person's Transactions Count...")

for name in X[:,nameOrigCol]:
	if nameCount.get(name,-1) == -1:
		nameOrig.append(name)

		nameCount[name] = 1

	else:
		nameCount[name] += 1
		namesWithMoreThanOneOccurrence.append(name)

for name in X[:,nameDestCol]:
	if nameCount.get(name,-1) == -1:
		nameDest.append(name)

		nameCount[name] = 1

	else:
		nameCount[name] += 1
		namesWithMoreThanOneOccurrence.append(name)

logging.warning("Count Identification Done")

logging.warning("Calculating Median ...")

countArr = []
count = 0
for attr, value in nameCount.items():
	if value>40:
		countArr.append(value)
		count += 1
median = np.median(countArr)

logging.warning(f"Median : {median}")

logging.warning("Filtering Data Based on Transactions Count...")
csv_golden_data = []

for i in range(X.shape[0]):
	if nameCount.get(X[i,3],-1) > 40 or nameCount.get(X[i,6],-1) > 40:
		csv_golden_data.append(X[i,:])

logging.warning("Filtering Done")

logging.warning("Storing Filtered Data in data_processed folder...")

with open("data_processed/filtered_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_golden_data)

logging.warning("Data is Stored")



