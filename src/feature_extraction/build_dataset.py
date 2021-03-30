import numpy as np
from tqdm import tqdm
import pandas as pd
from feature_extractor import path_to_feature_dict

"""First we extract features from all images for which the solar radiation values are valid"""

feature_records = []
count = 0
with open("../../Solais_Data/valid_img_paths.txt") as f:
    for path in f:
        count +=1
        record = path_to_feature_dict("../"+ path[:-1])
        feature_records.append(record)
        print(f"{count} images treated")

# Create a dataframe from records
features = pd.DataFrame(feature_records)
# Correct a feature default value
features[features.cloud_brightness == 0.] = 1.
# df.to_csv("../../processed_data/features.csv")

"""We merge the obtained dataset with the GHI dataset, and we build a clean final dataset"""

# Get solar radiation data
ghi = pd.read_csv("../../Solais_Data/ghi_data_5min.csv")

df = pd.concat([features, ghi], axis=1)
df["Timestamp"] = pd.to_datetime(df["Unnamed: 0"], format='%Y-%m-%d %H:%M:%S')
df.drop(columns=["Unnamed: 0"])
# Set a DateTime index
df.set_index("Timestamp",drop=True, inplace=True)

# Let's delete useless columns (ghi1 to ghi4), and extract ghi data that may be useful later :
del_col = ["Unnamed: 0", "ghi1", "ghi2", "ghi3", "ghi4", "ghi_mean", "mean/clearsky"]
df.drop(columns=del_col, inplace=True)

def clean_dataset(df):
    prev_len = len(df)
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    new_len = len(indices_to_keep)
    print(f"we dropped {(prev_len - new_len)/prev_len*100} % of the lines")
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df)

# Drop the 2 rows for which Kc == 0.
df = df[df.Kc != 0]

# Save the solar radiation features separately from the final dataset
ghi_col = ["clearsky_ghi", "clearsky_dni", "clearsky_dhi", "ghi_clipped"]
ghi_data = df[ghi_col]
ghi_data.to_csv("../../processed_data/ghi_dataset.csv", date_format='%Y-%m-%d %H:%M:%S')

# and exclude them from the final dataset
df.drop(columns=ghi_col, inplace=True)

# Save the final dataset
df.to_csv("../../processed_data/dataset_final.csv", date_format='%Y-%m-%d %H:%M:%S') 


