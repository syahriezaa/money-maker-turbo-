import os
import requests
import json

script_file = r"C:\Users\junaidi\.gemini\antigravity-ide\brain\517b9285-26e6-456c-bfb9-98b56efe2f5b\reddit_story_script.md"

if not os.path.exists(script_file):
    print(f"Error: Script file not found at {script_file}")
    exit(1)

with open(script_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

clean_paragraphs = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # Skip comments, markdown headers, dividers
    if line.startswith("#"):
        continue
    if line.startswith("---"):
        continue
    if line.startswith("##"):
        continue
    if "📌" in line:
        continue
    if line.startswith("1. \"My Wife") or line.startswith("2. \"I Found") or line.startswith("3. \"AITA"):
        continue
    if line.startswith("reddit story,") or line.startswith("- Pace:") or line.startswith("- Tone:") or line.startswith("- Pauses:") or line.startswith("- Duration"):
        continue
    clean_paragraphs.append(line)

story_text = "\n\n".join(clean_paragraphs)
paragraph_count = len(clean_paragraphs)

print(f"Parsed story: {paragraph_count} paragraphs found.")

payload = {
    "video_subject": f"Topic: I Discovered My Perfect Wife Was Living a Double Life\nContent Script:\n{story_text}",
    "video_aspect_ratio": "9:16",
    "voice_name": "chatterbox",
    "language": "en",
    "paragraph_number": paragraph_count,
    "local_steps": 12,
    "local_cfg": 7.5,
    "image_style": "realistic"
}

url = "http://localhost:8000/api/v1/videos"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print("FastAPI Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error connecting to backend server: {e}")
