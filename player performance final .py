#!/usr/bin/env python
# coding: utf-8

# In[17]:


import zipfile
import os
import pandas as pd

# Define file paths
zip_path = r"C:\Users\gurob\Desktop\recently_added_2_male_csv (2).zip"
extract_path = r"C:\Users\gurob\Desktop\extracted_csv_files"

# Extract ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List extracted CSV files
csv_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]
print(f" Extracted {len(csv_files)} CSV files: {csv_files}")


# In[19]:


csv_file_path = os.path.join(extract_path, "1444513.csv")  # Replace with desired file
df = pd.read_csv(csv_file_path)

# Display first few rows
print(df.head())


# In[ ]:




