import os
from googleapiclient.discovery import build
import pandas as pd
import random
from datetime import datetime, timezone
import json
from termcolor import colored
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
from youtube_transcript_api import YouTubeTranscriptApi
import waybackpy


def get_authenticated_service():
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=53286)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)


def get_popular_videos(api_key, region_code='GB', max_results=5, youtube=None):
    if not youtube:
        youtube = build('youtube', 'v3', developerKey=api_key)

    response = youtube.videos().list(
        part="snippet",
        chart='mostPopular',
        regionCode=region_code,
        maxResults=max_results
    ).execute()

    videos = {item['id']: item['snippet']['title'] for item in response.get('items', [])}
    return videos


def get_video_context(api_key, video_id, video_categories, youtube=None):
    if not youtube:
        youtube = build('youtube', 'v3', developerKey=api_key)
    
    response = youtube.videos().list(
        part="snippet,contentDetails",
        id=video_id
    ).execute()

    video = response['items'][0]['snippet']
    title = video['title']
    description = video['description']
    if 'tags' in video:
        tags = video['tags']
    else:
        tags = ""
    publish_time = video['publishedAt']
    obtain_time = datetime.now(timezone.utc)
    obtain_time = obtain_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    category = video_categories[video['categoryId']]

    thumbnails = video['thumbnails']
    sizes = ['maxres', 'standard', 'high', 'medium', 'default']
    # Get first available best size
    for size in sizes:
        if size in thumbnails:
            thumbnail = thumbnails[size]
            break

    # Get video captions/transcript - uses https://github.com/jdepoix/youtube-transcript-api
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    file_path = f'captions/captions_{video_id}.json'
    with open(file_path, 'w') as json_file:
        json.dump(transcript, json_file, indent=4)

    return [title, description, tags, publish_time, obtain_time, category, thumbnail]


def get_all_video_comments(api_key, video_id, max_comments=None, youtube=None):
    if not youtube:
        youtube = build('youtube', 'v3', developerKey=api_key)
    
    comments = []
    next_page_token = None
    
    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100, # API allows max 100 per request
            pageToken=next_page_token,
            textFormat='plainText',
        ).execute()
        
        for comment_thread in response.get('items', []):
            comment_id = comment_thread['snippet']['topLevelComment']['id']
            top_comment = comment_thread['snippet']['topLevelComment']['snippet']
            comment_text = top_comment['textDisplay']
            if len(comment_text.split()) < 3:
                continue
            publish_time = top_comment['publishedAt']
            obtain_time = datetime.now(timezone.utc)
            obtain_time = obtain_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            like_count = top_comment['likeCount']
            reply_count = comment_thread['snippet']['totalReplyCount']

            comment = [comment_id, video_id, comment_text, publish_time, obtain_time, like_count, reply_count]
            comments.append(comment)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    # Sort comments by the publish_time field
    comments.sort(key=lambda x: x[3])  # Sorting by the 4th item in the list, which is publish_time
    
    # Optionally trim the list to max_comments if needed
    if max_comments:
        comments = comments[:max_comments]
    
    return comments

def archive_webpage(video_id):
    url = "https://www.youtube.com/watch?v=" + video_id
    user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
    wayback = waybackpy.Url(url, user_agent) 
    archive = wayback.save()

comment_labels = pd.DataFrame(columns=['comment_id',
                                       'video_id',
                                       'comment_text',
                                       'comment_published_time',
                                       'comment_obtained_time',
                                       'comment_like_count',
                                       'comment_reply_count',
                                       'spam_without_context',
                                       'spam_with_context',
                                       'non_english'])

video_contexts = pd.DataFrame(columns=['video_id',
                                       'video_title',
                                       'video_description',
                                       'video_tags',
                                       'video_publish_time',
                                       'video_obtained_time',
                                       'video_category',
                                       'video_thumbnail',
                                       ])
video_contexts.set_index('video_id', inplace=True)
API_KEY = "AIzaSyB6qaMwoz9K9QXYE4es087YTiZFhbngyKo"
VIDEO_ID = "H6lQkgJ-jRo" # 161 comments
#VIDEO_ID = "89-GJHoqtiQ" # 1000 comments
#VIDEO_ID = "WkLTWmlTaJM" # 4000 comments
VIDEO_ID = "mKdjycj-7eE"
MAX_POPULAR_VIDEOS = 3
MAX_COMMENTS_PER_VIDEO = 1000

# Get video category mapping
video_categories_file = 'saved_data/video_categories.json'
with open(video_categories_file, 'r') as json_file:
    video_categories = json.load(json_file)

youtube = build('youtube', 'v3', developerKey=API_KEY)
#youtube = get_authenticated_service()

popular_videos = get_popular_videos(API_KEY, max_results=MAX_POPULAR_VIDEOS, youtube=youtube)

all_comments = []

try:
    #for video_id, video_title in popular_videos.items():
    for video_id in [VIDEO_ID]:
        # Get video context
        context = get_video_context(API_KEY, video_id, video_categories, youtube=youtube)
        video_contexts.loc[video_id] = context

        # Get video comments
        comments = get_all_video_comments(API_KEY, video_id, youtube=youtube, max_comments=1000)
        all_comments += comments

        # Archive webpage of video on web.archive.org - commented out for now because it takes a while to archive
        #archive_webpage(video_id)
    
    video_contexts.to_csv('video_contexts.csv')
    
    #
    # random.shuffle(all_comments)
    count = 1
    num_of_comments = len(all_comments)
    
    # First pass: No context shown

    for comment_data in all_comments:
        os.system('cls')
        print("\nFirst Pass - Only Comment content\n")
        print(f"{count}/{num_of_comments}\n")
        comment_text = comment_data[2]
        video_id = comment_data[0]
        print("Comment:")
        print(colored(comment_text, 'yellow'))
        spam = input("\n")
        if spam == "":
            new_row = comment_data + [0, "", 0]
        
        elif spam == "q":
            exit()
        
        elif spam == "n":
            new_row = comment_data + ["", "", 1]

        else:
            new_row = comment_data + [1, "", 0]
        comment_labels.loc[len(comment_labels)] = new_row
        count += 1

except KeyboardInterrupt:
    pass

try:
    # Second pass - with context
    count = 1
    num_of_comments = len(comment_labels[comment_labels['spam_without_context'] != ""])
    # Group comments by video_id to review them video by video
    grouped_comments = comment_labels.groupby('video_id')

    print("\nSecond Pass - With Video Context\n")
    for video_id, group in grouped_comments:
        video_info = video_contexts.loc[video_id]  

        for index, row in group.iterrows():

            # If comment was non-english or if comment was not labelled in first pass
            if row['spam_without_context'] == "":
                continue

            comment_text = row['comment_text']
            os.system('cls')
            print("Second Pass - With Context\n")
            print(f"{count}/{num_of_comments}\n")
            count += 1

            print("Video Description:")
            print(video_info['video_description'])
            print("\nVideo Title:")
            print(colored(video_info['video_title'], color='light_blue', attrs=['bold']))
            print("\nTags: ", colored(video_info['video_tags'], 'green'))
            print("Category: ", colored(video_info['video_category'], 'blue'))
            print()
            print("Like count:", colored(row['comment_like_count'], 'light_red'), 
                  ", Reply count:", colored(row['comment_reply_count'], 'light_red'))
            print("\nComment:")
            print(colored(comment_text, 'yellow'))
            spam_with_context = input("\n")

            if spam_with_context == "":
                comment_labels.at[index, 'spam_with_context'] = 0

            elif spam_with_context == "q":
                break

            else:
                comment_labels.at[index, 'spam_with_context'] = 1

    comment_labels.to_csv("comment_labels.csv")

except KeyboardInterrupt:
    print("Writing to csv...")
    comment_labels.to_csv("comment_labels.csv")
