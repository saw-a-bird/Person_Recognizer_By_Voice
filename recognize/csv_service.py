import os
import csv
from pandas import read_csv

DATASET_DIRECTORY = "/home/mahdi/Desktop/Python Notebook/speech_recognizer/recognize/csv_dir"
DATASET_VERSION = 2
DATASET_HEADERS = ['file_path', 'sound_rate', 'silence_duration', 'person_name']

def get_location_csv():
    return os.path.join(DATASET_DIRECTORY, "dataset-v"+str(DATASET_VERSION)+".csv")

def get_csv():
    return read_csv(get_location_csv())

def get_all_labels():
    df = get_csv()
    return df["person_name"].unique().tolist()

def get_label(i):
    all_labels = get_all_labels()
    return all_labels[i]

def create_if_needed():
    database_name = get_location_csv()
    
    writer = None
    database = None
    
    if os.path.exists(database_name) == False:
       # print("does not exists")
        database = open(database_name, "w+")
        writer = csv.writer(database)
        writer.writerow(DATASET_HEADERS) # write the header
    else:
      #  print("exist")
        database = open(database_name, "a")
        writer = csv.writer(database)
        
    return database, writer
  
          
def csv_add_new(name: str, records : list):
    database, writer = create_if_needed()
    for record in records:
        add = [record.get("filename"), record.get("sound_rate"), record.get("silence_duration"), name]
        if (len(add) == len(DATASET_HEADERS)):
            writer.writerow(add)
        
    database.close()

def csv_remove(name: str):
    database_name = get_location_csv()

    if os.path.exists(database_name) == True:
        df = read_csv(database_name)
        df = df.loc[ df["person_name"] != name ]
        df.to_csv(database_name, index=False)