import pandas as pd

csvs = ['saved_data/online_dataset/Youtube01-Psy.csv', 'saved_data/online_dataset/Youtube02-KatyPerry.csv',
        'saved_data/online_dataset/Youtube03-LMFAO.csv', 'saved_data/online_dataset/Youtube04-Eminem.csv',
        'saved_data/online_dataset/Youtube05-Shakira.csv']
video_ids = ['9bZkp7q19f0', 'CevxZvSJLk8', 'KQ6zr6kCPj8', 'uelHwf8o7_U', 'pRpeEdMmmQ0']

# Create an empty list to store the DataFrames
df_list = []

# Loop through each CSV file and read it into a DataFrame
for csv_file, video_id in zip(csvs, video_ids):
    df = pd.read_csv(csv_file)
    df['video_id'] = video_id
    df_list.append(df)

# Concatenate all the DataFrames vertically
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('saved_data/online_dataset/combined_data.csv', index=False)