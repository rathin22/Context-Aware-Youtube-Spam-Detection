from googleapiclient.discovery import build
import json

API_KEY = "AIzaSyB6qaMwoz9K9QXYE4es087YTiZFhbngyKo"
VIDEO_ID = "H6lQkgJ-jRo" # 161 comments

youtube = build('youtube', 'v3', developerKey=API_KEY)

response = youtube.videoCategories().list(
    part="snippet",
    regionCode="GB",
).execute()
print()
categories = {}
for item in response["items"]:
    print(f"{item['id']}: {item['snippet']['title']}")
    print()
    categories[item['id']] = item['snippet']['title']

file_path = 'saved data/video_categories.json'
with open(file_path, 'w') as json_file:
    json.dump(categories, json_file, indent=4)