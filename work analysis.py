### Step 1 - Import libraries and load data

import re
import pandas as pd

FILE = r"C:\Users\lod19\OneDrive\Desktop\The BIG project\report PII redacted.csv"
df = pd.read_csv(FILE)

print(df.columns)

### Step 2 - Keep relevant columns and rename them

keep = [ 
       'Reference',
       'Complaint - Patient Relations Team Management (First Received)',
       'Complaint Response Provided (--Due date)',
       'Complaint Response Provided (--Completion Date)',
       'Complaint Response Provided (Date Complaint Closed)',
       'Complaint - Patient Relations Team Management (PRM Handler)',
       'Complaint - Patient Relations Team Management (Responsible Hospital Exec Team (Main))',
       'Complaint - Patient Relations Team Management (Specialty (Main))',
       'Complaint - Patient Relations Team Management (Clinical Decision Group (CDG) (Main))',
       'Complaint - Patient Relations Team Management (Site of event (Main))',
       'Complaint - Patient Relations Team Management (Location (Main))',
       'Complaint - Patient Relations Team Management (Current stage)',
       'Complaint - Patient Relations Team Management (Complaint Complexity)',
       'Complaint - Patient Relations Team Management (Type of Complaint/PALS)',
       'Complaint - Patient Relations Team Management (Third party interest)',
       'Complaint - Patient Relations Team Management (Subjects (Complaint))',
       'Complaint - Patient Relations Team Management (Sub-Subjects (Complaint))',
       'Complaint - Patient Relations Team Management (Deadline Met)',
       'Complaint - Patient Relations Team Management (Outcome code (Complaint))',
       'Complaint Triage (Complaint Complexity)',
       'Complaint - Patient Relations Logging (Date of Birth (Complaint))',
       'Complaint - Patient Relations Team Management (Gender (KO41))',
       'Complaint - Patient Relations Team Management (Patient Ethnicity (KO41(A)))'
]

df = df.filter(items=keep)
print(df.columns)

rename_map = {
    'Complaint - Patient Relations Team Management (First Received)': 'First Received',
    'Complaint Response Provided (--Due date)': 'Due Date',
    'Complaint Response Provided (--Completion Date)': 'Completion Date',
    'Complaint Response Provided (Date Complaint Closed)': 'Date Complaint Closed',
    'Complaint - Patient Relations Team Management (PRM Handler)': 'PRM Handler',
    'Complaint - Patient Relations Team Management (Responsible Hospital Exec Team (Main))': 'Responsible Hospital Exec Team',
    'Complaint - Patient Relations Team Management (Specialty (Main))': 'Specialty',
    'Complaint - Patient Relations Team Management (Clinical Decision Group (CDG) (Main))': 'Clinical Decision Group',
    'Complaint - Patient Relations Team Management (Site of event (Main))': 'Site of Event',
    'Complaint - Patient Relations Team Management (Location (Main))': 'Location',
    'Complaint - Patient Relations Team Management (Current stage)': 'Current Stage',
    'Complaint - Patient Relations Team Management (Complaint Complexity)': 'Complaint Complexity',
    'Complaint - Patient Relations Team Management (Type of Complaint/PALS)': 'Type of Complaint/PALS',
    'Complaint - Patient Relations Team Management (Third party interest)': 'Third Party Interest',
    'Complaint - Patient Relations Team Management (Subjects (Complaint))': 'Subjects',
    'Complaint - Patient Relations Team Management (Sub-Subjects (Complaint))': 'Sub-Subjects',
    'Complaint - Patient Relations Team Management (Deadline Met)': 'Deadline Met',
    'Complaint - Patient Relations Team Management (Outcome code (Complaint))': 'Outcome Code',
    'Complaint Triage (Complaint Complexity)': 'Triage Complaint Complexity',
    'Complaint - Patient Relations Logging (Date of Birth (Complaint))': 'Date of Birth',
    'Complaint - Patient Relations Team Management (Gender (KO41))': 'Gender',
    'Complaint - Patient Relations Team Management (Patient Ethnicity (KO41(A)))': 'Patient Ethnicity'
}

df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
print(df.columns)

### Step 3 - Analyze data types and missing values
