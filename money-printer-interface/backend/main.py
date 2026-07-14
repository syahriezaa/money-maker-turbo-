import os
# Sanitize no_proxy to prevent httpx from crashing on IPv6 addresses (containing colons)
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import re
import sys
import uuid
import json
import time
import urllib.request
import base64
import shutil
import threading
import subprocess
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import urllib.parse
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
from moviepy.video.fx import Loop, Crop

app = FastAPI(title="MoneyPrinterTurbo Mock API", version="1.0.0")

# Enable CORS for all origins (frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"status": "ok", "message": "MoneyPrinterTurbo Mock API is running"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories
os.makedirs(STATIC_DIR, exist_ok=True)

class SettingsModel(BaseModel):
    llm_provider: str = "openai"
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    tts_provider: str = "edge-tts"
    voice_name: str = "en-US-GuyNeural"
    pexels_api_key: str = ""
    subtitles_enabled: bool = True
    subtitle_color: str = "#FFFFFF"
    subtitle_fontsize: int = 24
    output_dir: str = "./output"
    local_steps: int = 20
    local_cfg: float = 7.5
    local_seed: int = 1337
    local_negative_prompt: str = "low quality, worst quality, deformed, bad anatomy, bad hands, blurry, watermark, text, signature"

class VideoRequest(BaseModel):
    video_subject: str
    video_aspect_ratio: str = "9:16"
    voice_name: str = "en-US-GuyNeural"
    language: str = "en"
    paragraph_number: int = 2
    local_steps: Optional[int] = None
    local_cfg: Optional[float] = None
    local_seed: Optional[int] = None
    local_negative_prompt: Optional[str] = None
    image_style: Optional[str] = "gtav"
    character_prompt: Optional[str] = None

# DB connection helper
def get_db_conn(dbname="money_printer"):
    return psycopg2.connect(
        dbname=dbname,
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )

# DB initialization
def init_db():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost",
            port=5432
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'money_printer'")
        exists = cur.fetchone()
        if not exists:
            print("Database money_printer does not exist. Creating...")
            cur.execute("CREATE DATABASE money_printer")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error checking/creating database: {e}")

    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key VARCHAR PRIMARY KEY,
                value JSONB
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id VARCHAR PRIMARY KEY,
                subject TEXT,
                script TEXT,
                aspect_ratio VARCHAR(10),
                voice_name VARCHAR(50),
                language VARCHAR(10),
                paragraph_number INT,
                duration_seconds FLOAT,
                created_at FLOAT,
                status VARCHAR(20),
                progress INT,
                step VARCHAR(50),
                logs JSONB
            )
        """)
        
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS local_steps INT")
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS local_cfg FLOAT")
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS local_seed INT")
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS local_negative_prompt TEXT")
        # Kolom sub_progress untuk real-time tracking proses TTS/Music/Image
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS sub_progress JSONB DEFAULT NULL")
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS image_style VARCHAR(50) DEFAULT 'gtav'")
        # Tambahkan kolom database character_prompt TEXT pada migrasi startup tabel tasks
        cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS character_prompt TEXT")
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id VARCHAR PRIMARY KEY,
                title TEXT,
                prompt TEXT,
                duration FLOAT,
                aspect_ratio VARCHAR(10),
                download_url TEXT,
                created_at VARCHAR(50)
            )
        """)
        
        cur.execute("SELECT COUNT(*) FROM videos")
        if cur.fetchone()[0] == 0:
            print("Seeding mock videos...")
            seed_videos = [
                (
                    "vid_seed_1",
                    "5 Mind-Blowing Facts About Space",
                    "Create a short video about space facts that are hard to believe.",
                    45.5,
                    "9:16",
                    "/static/sample.mp4",
                    (datetime.now() - timedelta(hours=2)).isoformat()
                ),
                (
                    "vid_seed_2",
                    "The Secrets of Deep Sea Creatures",
                    "Deep sea monsters and their biological adaptations.",
                    60.0,
                    "16:9",
                    "/static/sample.mp4",
                    (datetime.now() - timedelta(hours=1)).isoformat()
                )
            ]
            for vid in seed_videos:
                cur.execute(
                    """
                    INSERT INTO videos (id, title, prompt, duration, aspect_ratio, download_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    vid
                )
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error initializing tables: {e}")

# Call init_db on startup
init_db()

def load_config() -> Dict[str, Any]:
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT value FROM config WHERE key = 'settings'")
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row["value"]
    except Exception as e:
        print(f"Error loading config: {e}")
        
    default_config = SettingsModel().model_dump() if hasattr(SettingsModel(), "model_dump") else SettingsModel().dict()
    save_config(default_config)
    return default_config

def save_config(config: Dict[str, Any]):
    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO config (key, value) VALUES ('settings', %s) ON CONFLICT (key) DO UPDATE SET value = %s",
            (json.dumps(config), json.dumps(config))
        )
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving config: {e}")

def ensure_mock_videos():
    target_path = os.path.join(STATIC_DIR, "sample.mp4")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
        return
    
    urls = [
        "https://www.w3schools.com/html/movie.mp4",
        "https://www.w3schools.com/html/mov_bbb.mp4",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    ]
    
    for url in urls:
        try:
            print(f"Downloading mock video from {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(target_path, "wb") as f:
                    f.write(response.read())
            print("Download successful.")
            return
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    # Fallback: create a tiny valid video structure if download fails.
    tiny_mp4_b64 = (
        "AAAAIGZ0eXBpc29tAAAAAGlzb21tcDQxAAAACHZyZWQAAAAIbW9vdgAAAGxtdmhkAAAAAM2d"
        "2pDNndqQAAAfQAAAA+gAAAEAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAA"
        "AAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAQAAAB1dHJhawAAAFx0a2hkAAAAA82d"
        "2pDNndqQAAAAAQAAAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAA"
        "AAAAAAAAAAAAAAABAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAA+gAAAAAABAAAAAABo"
        "bWRpYQAAACBtZGhkAAAAAM2d2pDNndqQAAAfQAAAA+gAAAAAAAAAIWhkbHIAAAAAAAAAAHZp"
        "ZGVAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAAAVW1pbmYAAAAUdm1oZAAAAAEAAAAAAAAA"
        "AAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAE0c3RibAAAALBzdHNkAAAA"
        "AAAAAAEAAAClYXZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAQABAAgAAAAIAAAAEAAAAA"
        "ABhjb2xybmN4YwAEAAQAAgAAAAAAYmF2Y0MBQQA5/+EAGGdkADkAsCtcBAQBAQCAgAAAAwCA"
        "AAAeQAAY8hA=AAAAHHV1aWRraWR4AAAAAAAAAAAAAAAAYmFjb24AAAAIc3R0cwAAAAAAAAAB"
        "AAAAAQAAAD6AAAAAc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAFHN0c3oAAAAAAAAAAAAA"
        "AAABAAAADHN0Y28AAAAAAAAAAQAAADw="
    )
    try:
        with open(target_path, "wb") as f:
            f.write(base64.b64decode(tiny_mp4_b64))
        print("Fallback: Created a base64 encoded tiny MP4.")
    except Exception as e:
        print(f"Fallback creation failed: {e}")
        with open(target_path, "wb") as f:
            f.write(b"\x00" * 1024 * 1024)

# Run seeding on startup
ensure_mock_videos()

# Helper to convert hex color to RGB tuple
def hex_to_rgb(hex_str: str) -> tuple:
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)

# Helper to find a font on Windows, falling back to default
def get_font(font_size: int) -> ImageFont.ImageFont:
    font_paths = [
        r"C:\Windows\Fonts\segoeuib.ttf",  # Segoe UI Bold
        r"C:\Windows\Fonts\segoeui.ttf",   # Segoe UI
        r"C:\Windows\Fonts\arialbd.ttf",   # Arial Bold
        r"C:\Windows\Fonts\arial.ttf",     # Arial
        r"C:\Windows\Fonts\calibri.ttf"    # Calibri
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

# Wrap text to fit width
def wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        # textbbox returns (x0, y0, x1, y1)
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines

# Create dark vertical gradient image
def create_dark_gradient(width: int, height: int, color1=(30, 30, 47), color2=(8, 8, 18)) -> Image.Image:
    base = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(base)
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line((0, y, width, y), fill=(r, g, b))
    return base

# Draw centered wrapped text on background (e.g. for slideshow)
def draw_wrapped_text_centered(image: Image.Image, text: str, font: ImageFont.ImageFont, max_width: int, text_color=(255, 255, 255), shadow_color=(0, 0, 0)):
    draw = ImageDraw.Draw(image)
    lines = wrap_text(text, draw, font, max_width)
    
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
        
    line_spacing = int(font.size * 0.3)
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    
    y = (image.height - total_height) // 2
    for i, line in enumerate(lines):
        line_w = line_widths[i]
        line_h = line_heights[i]
        x = (image.width - line_w) // 2
        
        # Shadow
        draw.text((x + 2, y + 2), line, font=font, fill=shadow_color)
        # Main text
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_h + line_spacing

# Overlay subtitle on frame (for Pexels video overlays)
def overlay_subtitle_on_frame(frame: np.ndarray, text: str, font: ImageFont.ImageFont, text_color: tuple) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    lines = wrap_text(text, draw, font, max_width=int(w * 0.85))
    
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
        
    line_spacing = int(font.size * 0.3)
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    
    # Place text in the lower third (e.g. 75% down)
    y = int(h * 0.75) - (total_height // 2)
    for i, line in enumerate(lines):
        line_w = line_widths[i]
        line_h = line_heights[i]
        x = (w - line_w) // 2
        
        # Shadow
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
        # Main text
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_h + line_spacing
        
    return np.array(img)

# Crop and resize helper
def resize_and_crop(clip, target_w: int, target_h: int):
    clip_w, clip_h = clip.size
    scale = max(target_w / clip_w, target_h / clip_h)
    new_w = int(clip_w * scale)
    new_h = int(clip_h * scale)
    
    resized_clip = clip.resized((new_w, new_h))
    
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    
    return resized_clip.with_effects([Crop(x1=x1, y1=y1, x2=x2, y2=y2)])

# Loop helper
def loop_clip(clip, target_duration: float):
    return clip.with_effects([Loop(duration=target_duration)])

# Clean subject search query
def clean_subject_for_search(subject: str) -> str:
    if "Topic:" in subject:
        parts = subject.split("Topic:")
        if len(parts) > 1:
            subject = parts[1]
    if "Content Script:" in subject:
        subject = subject.split("Content Script:")[0]
    subject = subject.replace(".", "").replace(":", "").strip()
    return subject if subject else "nature"

# Pexels Video Clip Sourcing
def get_pexels_video(query: str, api_key: str, aspect_ratio: str, target_duration: float, task_id: str) -> Optional[str]:
    orientation = "portrait" if aspect_ratio == "9:16" else "landscape"
    headers = {"Authorization": api_key}
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://api.pexels.com/videos/search?query={encoded_query}&orientation={orientation}&per_page=5"
    try:
        print(f"Pexels Search: {url}")
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        videos = res.json().get("videos", [])
        
        if not videos:
            print("No portrait/landscape video found. Retrying without orientation filter...")
            url_no_orient = f"https://api.pexels.com/videos/search?query={encoded_query}&per_page=5"
            res = requests.get(url_no_orient, headers=headers, timeout=15)
            res.raise_for_status()
            videos = res.json().get("videos", [])
            
        if not videos:
            print("No videos found on Pexels for query:", query)
            return None
            
        for video in videos:
            files = video.get("video_files", [])
            hd_file = None
            sd_file = None
            for f in files:
                if f.get("file_type") == "video/mp4":
                    quality = f.get("quality", "").lower()
                    if quality == "hd":
                        hd_file = f
                    elif quality == "sd":
                        sd_file = f
            selected_file = hd_file or sd_file or (files[0] if files else None)
            if selected_file and selected_file.get("link"):
                video_url = selected_file["link"]
                temp_filename = os.path.join(STATIC_DIR, f"temp_{task_id}.mp4")
                print(f"Downloading Pexels video: {video_url} -> {temp_filename}")
                
                req = urllib.request.Request(
                    video_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(temp_filename, "wb") as out_file:
                        out_file.write(response.read())
                print("Download complete.")
                return temp_filename
    except Exception as e:
        print(f"Error fetching/downloading Pexels video: {e}")
    return None

# Mount the static directory to serve video files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def cleanup_task_files(task_id: str):
    import glob
    patterns = [
        f"audio_{task_id}.wav",
        f"music_{task_id}.wav",
        f"mixed_{task_id}.wav",
        f"video_{task_id}.mp4",
        f"scene_{task_id}_*.png",
        f"scene_{task_id}_*.mp4",
    ]
    for pattern in patterns:
        files = glob.glob(os.path.join(STATIC_DIR, pattern))
        for f in files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Cleaned up file: {f}")
            except Exception as e:
                print(f"Error deleting file {f}: {e}")

def process_task_background(task_id: str, subject: str, aspect_ratio: str, voice_name: str, language: str, paragraph_number: int, duration_seconds: float, tts_provider: str):
    print(f"[DEBUG] process_task_background STARTED for task {task_id}", flush=True)
    def log_line(category: str, message: str):
        log_time = datetime.now()
        return f"[{log_time.strftime('%Y-%m-%d %H:%M:%S')}] [{category}] {message}"
        
    logs = [log_line("LLM Scripting", "Initializing video generation task...")]
    
    def update_db(progress: int, step: str, new_log_cat: str = None, new_log_msg: str = None, status: str = "processing"):
        if new_log_cat and new_log_msg:
            logs.append(log_line(new_log_cat, new_log_msg))
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET progress = %s, step = %s, logs = %s, status = %s WHERE id = %s",
                (progress, step, json.dumps(logs), status, task_id)
            )
            cur.close()
            conn.close()
        except Exception as db_e:
            print(f"Failed to update task state in DB: {db_e}")

    # ── Helper: simpan sub_progress ke database ───────────────────────────────
    def update_sub_progress(data: dict):
        """Simpan data sub-progress (step, speed, ETA) ke kolom sub_progress di DB."""
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET sub_progress = %s WHERE id = %s",
                (json.dumps(data), task_id)
            )
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Failed to update sub_progress: {e}")

    # ── Helper: clear sub_progress saat task selesai atau gagal ───────────────
    def clear_sub_progress():
        """Hapus data sub-progress setelah task selesai atau gagal."""
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET sub_progress = NULL WHERE id = %s",
                (task_id,)
            )
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Failed to clear sub_progress: {e}")

    # ── Pola regex untuk mem-parse output tqdm ────────────────────────────────
    # Contoh baris: " 45%|████      | 450/1000 [00:15<00:18, 28.50it/s]"
    def run_subprocess_with_progress(cmd: list, phase: str, segment: int = None, total_segments: int = None) -> tuple:
        """
        Jalankan subprocess dan baca output per-chunk (bukan per-baris) agar
        tqdm yang menggunakan \\r (carriage return) dapat di-parse secara real-time.

        Returns:
            tuple: (stdout_combined: str, returncode: int)
        """
        import time as _time_local
        all_output = []
        last_db_write = 0.0  # throttle: max 1x update per detik
        _tqdm_re = re.compile(r'(\d+)%.*?(\d+)/(\d+).*?([\d.]+)it/s')

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # gabung stderr ke stdout agar tqdm terbaca
                text=True,
                bufsize=0  # unbuffered — terima data sesegera mungkin
            )
            partial = ""
            while True:
                chunk = proc.stdout.read(256)
                if not chunk:
                    break
                all_output.append(chunk)
                partial += chunk
                # Split pada \r atau \n — tqdm pakai \r, log normal pakai \n
                segments_raw = partial.replace('\r', '\n').split('\n')
                partial = segments_raw[-1]  # simpan sisa yang belum lengkap
                for seg in segments_raw[:-1]:
                    seg = seg.strip()
                    if not seg:
                        continue
                    m = _tqdm_re.search(seg)
                    if m:
                        now = _time_local.time()
                        if now - last_db_write >= 1.0:  # throttle 1 detik
                            last_db_write = now
                            pct = float(m.group(1))
                            step_val = int(m.group(2))
                            total_steps = int(m.group(3))
                            speed = float(m.group(4))
                            remaining = total_steps - step_val
                            eta_seconds = int(remaining / speed) if speed > 0 else None
                            update_sub_progress({
                                "phase": phase,
                                "segment": segment,
                                "total_segments": total_segments,
                                "pct": pct,
                                "step": step_val,
                                "total_steps": total_steps,
                                "speed": speed,
                                "eta_seconds": max(0, eta_seconds) if eta_seconds is not None else None
                            })
            proc.wait()
            return "".join(all_output), proc.returncode
        except Exception as e:
            print(f"run_subprocess_with_progress error: {e}")
            return "", 1

            
    def check_cancelled():
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT status FROM tasks WHERE id = %s", (task_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and row[0] and row[0].lower() == 'cancelled':
                print(f"Task {task_id} detected as cancelled. Cleaning up and exiting thread.")
                logs.append(log_line("Cancellation", "Task execution cancelled by user. Cleaning up files..."))
                try:
                    conn = get_db_conn()
                    conn.autocommit = True
                    cur = conn.cursor()
                    cur.execute("UPDATE tasks SET logs = %s WHERE id = %s", (json.dumps(logs), task_id))
                    cur.close()
                    conn.close()
                except Exception as db_e:
                    print(f"Failed to update logs on cancellation: {db_e}")
                cleanup_task_files(task_id)
                return True
        except Exception as e:
            print(f"Error checking cancellation status: {e}")
        return False
    
    try:
        # Mengambil konfigurasi pembuatan gambar lokal yang spesifik untuk tugas ini
        t_local_steps = None
        t_local_cfg = None
        t_local_seed = None
        t_local_neg = None
        t_image_style = "gtav"
        character_prompt = None
        try:
            conn = get_db_conn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT local_steps, local_cfg, local_seed, local_negative_prompt, image_style, character_prompt FROM tasks WHERE id = %s", (task_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                t_local_steps = row.get("local_steps")
                t_local_cfg = row.get("local_cfg")
                t_local_seed = row.get("local_seed")
                t_local_neg = row.get("local_negative_prompt")
                t_image_style = row.get("image_style") or "gtav"
                character_prompt = row.get("character_prompt")
        except Exception as db_err:
            print(f"Error querying task settings from database: {db_err}")

        global_config = load_config()
        local_steps = t_local_steps if t_local_steps is not None else global_config.get("local_steps", 20)
        local_cfg = t_local_cfg if t_local_cfg is not None else global_config.get("local_cfg", 7.5)
        local_seed = t_local_seed if t_local_seed is not None else global_config.get("local_seed", 1337)
        local_negative_prompt = t_local_neg if t_local_neg is not None else global_config.get("local_negative_prompt", "low quality, worst quality, deformed, bad anatomy, bad hands, blurry, watermark, text, signature")
        image_style = t_image_style

        is_local = (tts_provider == "local-chatterbox")
        sleep_step = (duration_seconds / 6.0) if not is_local else 0.5
        
        update_db(5, "LLM Scripting", "LLM Scripting", f"Generating script using LLM with prompt: '{subject}'")
        time.sleep(sleep_step)
        
        # Script generation logic:
        # Check if subject has Topic and Content Script
        actual_subject = subject
        script_text = ""
        if "Topic:" in subject and "Content Script:" in subject:
            try:
                parts = subject.split("Content Script:")
                script_text = parts[1].strip()
                topic_part = parts[0].replace("Topic:", "").strip()
                if topic_part.endswith("."):
                    topic_part = topic_part[:-1].strip()
                actual_subject = topic_part
            except Exception:
                pass
                
        if not script_text:
            # Let's generate it
            script_text = generate_simulated_script(actual_subject, language, paragraph_number)
            
        # Update database with the final script
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("UPDATE tasks SET script = %s WHERE id = %s", (script_text, task_id))
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Failed to update task script: {e}")
        
        update_db(15, "LLM Scripting", "LLM Scripting", f"Script generated successfully! Paragraph count: {paragraph_number}")
        time.sleep(sleep_step)
        
        # Voice Synthesis Step
        if check_cancelled():
            return
        update_db(25, "Voice Synthesis", "Voice Synthesis", f"Starting Voice Synthesis using voice: '{voice_name}' (Provider: {tts_provider})")
        
        wav_path = os.path.join(STATIC_DIR, f"audio_{task_id}.wav")
        dest_video_path = os.path.join(STATIC_DIR, f"video_{task_id}.mp4")
        
        audio_duration = None
        if is_local:
            update_db(35, "Voice Synthesis", "Voice Synthesis", "Running Chatterbox Turbo TTS engine locally...")
            
            # Use current backend python interpreter
            interpreter = sys.executable
            script_path = os.path.join(BASE_DIR, "generate_tts_local.py")
            
            cmd = [interpreter, script_path, script_text, wav_path, language, voice_name]
            print(f"Running local TTS subprocess: {' '.join(cmd)}")
            # Gunakan Popen agar progress tqdm bisa dibaca real-time
            stdout_out, returncode = run_subprocess_with_progress(
                cmd,
                phase="Sintesis Suara",
                segment=None,
                total_segments=None
            )
            # Bersihkan sub_progress TTS sebelum lanjut ke tahap berikutnya
            clear_sub_progress()

            if returncode != 0:
                raise Exception(f"Chatterbox TTS subprocess failed: {stdout_out}")

            update_db(45, "Voice Synthesis", "Voice Synthesis", f"Audio file synthesized locally. Saved to static/audio_{task_id}.wav")
            
            # Load audio to get its duration
            audio_clip = AudioFileClip(wav_path)
            audio_duration = audio_clip.duration
            audio_clip.close()
        else:
            update_db(35, "Voice Synthesis", "Voice Synthesis", f"Running Edge-TTS engine locally with voice '{voice_name}'...")
            try:
                edge_tts_exe = os.path.join(os.path.dirname(sys.executable), "edge-tts")
                cmd = [edge_tts_exe, "--voice", voice_name, "--text", script_text, "--write-media", wav_path]
                print(f"Running Edge-TTS subprocess: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    update_db(45, "Voice Synthesis", "Voice Synthesis", f"Audio file synthesized via Edge-TTS. Saved to static/audio_{task_id}.wav")
                    audio_clip = AudioFileClip(wav_path)
                    audio_duration = audio_clip.duration
                    audio_clip.close()
                else:
                    raise Exception(f"Edge-TTS CLI failed: {result.stderr}")
            except Exception as e:
                print(f"Edge-TTS failed: {e}. Falling back to simulated audio duration.")
                update_db(45, "Voice Synthesis", "Voice Synthesis", f"Edge-TTS failed: {e}. Using simulated time.")
                audio_duration = duration_seconds
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
            
        # ─── Local AI Music Generation + Audio Ducking ────────────────────────────
        if check_cancelled():
            return
        # Runs for all providers but only if local benchmark python is available.
        mixed_audio_path = None
        if os.path.exists(wav_path) and audio_duration:
            try:
                update_db(48, "AI Music", "AI Music", "Generating AI background music with MusicGen locally...")
                # Use current backend python interpreter
                local_benchmark_python_music = sys.executable

                music_script = os.path.join(BASE_DIR, "generate_music_local.py")
                raw_music_path = os.path.join(STATIC_DIR, f"music_{task_id}.wav")
                # Infer music mood from subject
                music_prompt = f"ambient cinematic background music for a video about {actual_subject}, no vocals"
                music_duration = min(30, int(audio_duration) + 4)

                music_cmd = [local_benchmark_python_music, music_script, music_prompt, raw_music_path, str(music_duration)]
                # Gunakan Popen agar progress tqdm MusicGen bisa dibaca real-time
                music_stdout, music_returncode = run_subprocess_with_progress(
                    music_cmd,
                    phase="Generasi Musik AI",
                    segment=None,
                    total_segments=None
                )
                # Bersihkan sub_progress music sebelum lanjut ke mixing
                clear_sub_progress()
                # Buat objek kompatibel dengan kode lama yang cek returncode dan path
                class _MusicResult:
                    def __init__(self, rc, out):
                        self.returncode = rc
                        self.stderr = out
                music_result = _MusicResult(music_returncode, music_stdout)
                if music_result.returncode == 0 and os.path.exists(raw_music_path):
                    update_db(49, "AI Music", "AI Music", "Music generated. Ducking audio levels dynamically...")
                    import numpy as np
                    from scipy.io.wavfile import read as wav_read, write as wav_write

                    # Load audio files
                    vo_sr, vo_data = wav_read(wav_path)
                    mu_sr, mu_data = wav_read(raw_music_path)

                    # Convert to float32 normalized to [-1.0, 1.0] if they are int16
                    if vo_data.dtype == np.int16:
                        vo_data = vo_data.astype(np.float32) / 32768.0
                    else:
                        vo_data = vo_data.astype(np.float32)

                    if mu_data.dtype == np.int16:
                        mu_data = mu_data.astype(np.float32) / 32768.0
                    else:
                        mu_data = mu_data.astype(np.float32)

                    # Make mono by averaging channels
                    if len(vo_data.shape) > 1:
                        vo_mono = vo_data.mean(axis=1)
                    else:
                        vo_mono = vo_data

                    if len(mu_data.shape) > 1:
                        mu_mono = mu_data.mean(axis=1)
                    else:
                        mu_mono = mu_data

                    # Resample music using linear interpolation if needed
                    if mu_sr != vo_sr:
                        num_samples = int(len(mu_mono) * vo_sr / mu_sr)
                        mu_mono = np.interp(
                            np.linspace(0, len(mu_mono), num_samples, endpoint=False),
                            np.arange(len(mu_mono)),
                            mu_mono
                        )

                    # Normalize peaks to 0.85
                    vo_peak = np.abs(vo_mono).max()
                    if vo_peak > 0:
                        vo_mono = vo_mono / vo_peak * 0.85

                    mu_peak = np.abs(mu_mono).max()
                    if mu_peak > 0:
                        mu_mono = mu_mono / mu_peak * 0.85

                    # Loop or trim music to match voiceover length
                    vo_len = len(vo_mono)
                    if len(mu_mono) < vo_len:
                        repeats = (vo_len // len(mu_mono)) + 1
                        mu_mono = np.tile(mu_mono, repeats)
                    mu_mono = mu_mono[:vo_len]

                    # VAD envelope: compute rolling RMS of voiceover
                    window = vo_sr // 10  # 100ms window
                    hop = window // 4
                    rms_len = (vo_len + hop - 1) // hop
                    rms = np.zeros(rms_len, dtype=np.float32)
                    
                    for i in range(rms_len):
                        start_idx = max(0, i * hop - window // 2)
                        end_idx = min(vo_len, i * hop + window // 2)
                        segment = vo_mono[start_idx:end_idx]
                        if len(segment) > 0:
                            rms[i] = np.sqrt(np.mean(segment ** 2))

                    # Build sample-level duck mask
                    duck_mask = np.zeros(vo_len, dtype=np.float32)
                    for i, r in enumerate(rms):
                        start_idx = i * hop
                        end_idx = min(start_idx + hop, vo_len)
                        duck_level = 0.30 if r > 0.01 else 0.70
                        duck_mask[start_idx:end_idx] = duck_level

                    # Smooth duck mask with moving average
                    smooth_w = vo_sr // 5  # 200ms
                    if smooth_w > 1:
                        kernel = np.ones(smooth_w) / smooth_w
                        duck_mask_smooth = np.convolve(duck_mask, kernel, mode='same')
                    else:
                        duck_mask_smooth = duck_mask

                    ducked_music = mu_mono * duck_mask_smooth
                    mixed = vo_mono * 0.70 + ducked_music
                    mixed = np.clip(mixed, -1.0, 1.0)

                    # Save as stereo int16 wav file (which is standard and compatible)
                    mixed_stereo = np.column_stack((mixed, mixed))
                    mixed_stereo_int16 = (mixed_stereo * 32767.0).astype(np.int16)

                    mixed_audio_path = os.path.join(STATIC_DIR, f"mixed_{task_id}.wav")
                    wav_write(mixed_audio_path, vo_sr, mixed_stereo_int16)
                    update_db(50, "AI Music", "AI Music", "Background music mixed with voiceover (smart ducking applied).")
                    # Clean up raw music file
                    try:
                        os.remove(raw_music_path)
                    except Exception:
                        pass
                else:
                    update_db(50, "Sourcing Media", "Sourcing Media", f"MusicGen failed (exit {music_result.returncode}). Continuing without background music.")
            except Exception as music_err:
                print(f"Local music generation/ducking failed: {music_err}")
                update_db(50, "Sourcing Media", "Sourcing Media", f"Music step skipped: {music_err}")
        else:
            update_db(50, "Sourcing Media", "Sourcing Media", "Searching for relevant background media from stock library...")

        # ─── Determine aspect ratio resolutions ───────────────────────────────────
        if aspect_ratio == "16:9":
            target_w, target_h = 1280, 720
        else:
            target_w, target_h = 720, 1280
            
        config = load_config()
        pexels_key = config.get("pexels_api_key", "").strip()
        subtitles_enabled = config.get("subtitles_enabled", True)
        subtitle_color_hex = config.get("subtitle_color", "#FFFFFF")
        subtitle_fontsize = config.get("subtitle_fontsize", 24)
        
        subtitle_color = hex_to_rgb(subtitle_color_hex)
        # ─── Local AI Image Generation (SDXL Turbo) per scene ────────────────────
        if check_cancelled():
            return
        ai_scene_clips = []
        if is_local:
            try:
                update_db(52, "AI Visuals", "AI Visuals", "Generating AI scene images with SDXL Turbo locally...")
                # Use current backend python interpreter
                local_benchmark_python_img = sys.executable

                img_gen_script = os.path.join(BASE_DIR, "generate_images_local.py")
                cam_motion_script = os.path.join(BASE_DIR, "apply_camera_motion.py")

                # Split script into visual scenes (sentences) and group them into slides of at least 15 words (~6-7 seconds per scene)
                raw_sents = re.split(r'(?<=[.!?])\s+', script_text.replace("\n\n", " ").replace("\n", " "))
                sents = [s.strip() for s in raw_sents if s.strip()]
                
                scene_paragraphs = []
                current_scene = []
                current_word_count = 0
                for sent in sents:
                    words_in_sent = len(sent.split())
                    current_scene.append(sent)
                    current_word_count += words_in_sent
                    if current_word_count >= 15:
                        scene_paragraphs.append(" ".join(current_scene))
                        current_scene = []
                        current_word_count = 0
                if current_scene:
                    if scene_paragraphs:
                        scene_paragraphs[-1] += " " + " ".join(current_scene)
                    else:
                        scene_paragraphs.append(" ".join(current_scene))
                        
                if not scene_paragraphs:
                    scene_paragraphs = [actual_subject]

                scene_duration = (audio_duration or duration_seconds) / max(len(scene_paragraphs), 1)

                for i, scene_text in enumerate(scene_paragraphs):
                    img_path = os.path.join(STATIC_DIR, f"scene_{task_id}_{i}.png")
                    scene_clip_path = os.path.join(STATIC_DIR, f"scene_{task_id}_{i}.mp4")

                    # Buat prompt visual dari adegan menggunakan fungsi terpusat
                    scene_prompt = generate_scene_image_prompt(scene_text, actual_subject, image_style, character_prompt)

                    res_arg = "512x896" if aspect_ratio == "9:16" else ("896x512" if aspect_ratio == "16:9" else "512x512")
                    # 1. Generate image
                    img_cmd = [
                        local_benchmark_python_img, 
                        img_gen_script, 
                        scene_prompt, 
                        img_path,
                        "--resolution", res_arg,
                        "--steps", str(local_steps),
                        "--cfg", str(local_cfg),
                        "--seed", str(local_seed),
                        "--negative-prompt", local_negative_prompt,
                        "--style", image_style
                    ]
                    # Gunakan Popen agar progress tqdm SDXL bisa dibaca real-time per scene
                    img_stdout, img_returncode = run_subprocess_with_progress(
                        img_cmd,
                        phase="Generasi Gambar AI",
                        segment=i + 1,
                        total_segments=len(scene_paragraphs)
                    )
                    # Bersihkan sub_progress image scene ini sebelum lanjut
                    clear_sub_progress()

                    if img_returncode == 0 and os.path.exists(img_path):
                        update_db(54 + i, "AI Visuals", "AI Visuals", f"Scene {i+1}/{len(scene_paragraphs)} image generated. Applying camera motion...")
                        # 2. Apply Ken Burns camera motion
                        cam_cmd = [local_benchmark_python_img, cam_motion_script, img_path, scene_clip_path, str(round(scene_duration, 2))]
                        cam_stdout, cam_returncode = run_subprocess_with_progress(
                            cam_cmd,
                            phase="Animasi Kamera",
                            segment=i + 1,
                            total_segments=len(scene_paragraphs)
                        )
                        clear_sub_progress()
                        if cam_returncode == 0 and os.path.exists(scene_clip_path):
                            ai_scene_clips.append(scene_clip_path)
                        else:
                            print(f"Camera motion failed for scene {i}: {cam_stdout}")
                    else:
                        print(f"Image generation failed for scene {i}: {img_stdout}")

                if ai_scene_clips:
                    update_db(60, "AI Visuals", "AI Visuals", f"{len(ai_scene_clips)}/{len(scene_paragraphs)} AI animated scenes ready.")
                else:
                    update_db(60, "Sourcing Media", "Sourcing Media", "AI image generation produced no usable scenes. Falling back to Pexels/slideshow.")
            except Exception as img_err:
                print(f"Local AI image generation failed: {img_err}")
                update_db(60, "Sourcing Media", "Sourcing Media", f"AI Visuals skipped: {img_err}. Using standard media.")

        temp_video_clip_path = None
        search_query = clean_subject_for_search(actual_subject)
        if not ai_scene_clips:
            if pexels_key:
                update_db(55, "Sourcing Media", "Sourcing Media", f"Pexels key configured. Searching for '{search_query}'...")
                temp_video_clip_path = get_pexels_video(search_query, pexels_key, aspect_ratio, audio_duration, task_id)
                if temp_video_clip_path:
                    update_db(60, "Sourcing Media", "Sourcing Media", "Pexels video clip downloaded successfully.")
                else:
                    update_db(60, "Sourcing Media", "Sourcing Media", "Pexels video not found or download failed. Falling back to slideshow.")
            else:
                time.sleep(sleep_step)
                update_db(60, "Sourcing Media", "Sourcing Media", "Pexels key not configured. Using slideshow generator.")
            
        # Rendering Video step
        if check_cancelled():
            return
        update_db(70, "Rendering Video", "Rendering Video", "Composing video timeline...")
        
        # Split script into paragraphs for slideshow/overlays
        paragraphs_list = [p.strip() for p in script_text.split("\n\n") if p.strip()]
        if not paragraphs_list:
            paragraphs_list = [p.strip() for p in script_text.split("\n") if p.strip()]
        if not paragraphs_list:
            paragraphs_list = [actual_subject]
            
        slide_duration = audio_duration / len(paragraphs_list)
        
        video_clip = None
        try:
            if temp_video_clip_path and os.path.exists(temp_video_clip_path):
                # We have a downloaded Pexels video
                bg_clip = VideoFileClip(temp_video_clip_path)
                # Crop and resize
                bg_clip = resize_and_crop(bg_clip, target_w, target_h)
                # Loop to match audio_duration
                bg_clip = loop_clip(bg_clip, audio_duration)
                
                # Apply subtitles if enabled
                if subtitles_enabled:
                    segments = []
                    font = get_font(subtitle_fontsize)
                    for i, para in enumerate(paragraphs_list):
                        t_start = i * slide_duration
                        t_end = min((i + 1) * slide_duration, audio_duration)
                        sub = bg_clip.subclipped(t_start, t_end)
                        sub = sub.image_transform(lambda frame, text=para, font=font, color=subtitle_color: overlay_subtitle_on_frame(frame, text, font, color))
                        segments.append(sub)
                    video_clip = concatenate_videoclips(segments, method="compose")
                else:
                    video_clip = bg_clip
            elif ai_scene_clips:
                # ── Use AI-generated animated scene clips ──
                from moviepy import VideoFileClip as _VFC
                raw_scene_clips = []
                for scp in ai_scene_clips:
                    sc = _VFC(scp)
                    # Resize each scene to target dimensions
                    sc = resize_and_crop(sc, target_w, target_h)
                    raw_scene_clips.append(sc)
                video_clip = concatenate_videoclips(raw_scene_clips, method="compose")
                # Loop to match full audio duration if shorter
                if video_clip.duration < (audio_duration or duration_seconds):
                    video_clip = loop_clip(video_clip, audio_duration or duration_seconds)
            else:
                # Generate Pillow slideshow fallback
                slides = []
                font = get_font(subtitle_fontsize if subtitle_fontsize else 32)
                for para in paragraphs_list:
                    img = create_dark_gradient(target_w, target_h)
                    draw_wrapped_text_centered(img, para, font, max_width=int(target_w * 0.85))
                    slide_clip = ImageClip(np.array(img)).with_duration(slide_duration)
                    slides.append(slide_clip)
                video_clip = concatenate_videoclips(slides, method="compose")
                
            # Mux with audio: prefer mixed (ducked) audio, then raw voiceover, then silent
            mixed_exists = mixed_audio_path and os.path.exists(mixed_audio_path)
            vo_exists = os.path.exists(wav_path)
            has_audio = mixed_exists or vo_exists
            if has_audio:
                update_db(80, "Rendering Video", "Rendering Video", "Muxing audio track (with AI music backing)..." if mixed_exists else "Muxing voiceover track...")
                chosen_audio = mixed_audio_path if mixed_exists else wav_path
                audio_clip = AudioFileClip(chosen_audio)
                final_clip = video_clip.with_audio(audio_clip)
                final_clip = final_clip.with_duration(audio_duration)
            else:
                final_clip = video_clip
                
            final_clip = final_clip.with_fps(24)
            
            print(f"Writing final video to {dest_video_path}")
            
            final_clip.write_videofile(
                dest_video_path,
                codec="libx264",
                audio_codec="aac" if has_audio else None,
                audio=has_audio,
                threads=1,
                logger=None
            )
            
            # Close clips
            final_clip.close()
            if has_audio:
                audio_clip.close()
            video_clip.close()
            
            # Clean up temp downloaded clip if exists
            if temp_video_clip_path and os.path.exists(temp_video_clip_path):
                try:
                    os.remove(temp_video_clip_path)
                except Exception:
                    pass
            # Clean up AI scene clips and images
            for scp in ai_scene_clips:
                try:
                    os.remove(scp)
                except Exception:
                    pass
            for i in range(len(ai_scene_clips)):
                img_p = os.path.join(STATIC_DIR, f"scene_{task_id}_{i}.png")
                try:
                    os.remove(img_p)
                except Exception:
                    pass
            # Clean up mixed audio if exists
            if mixed_audio_path and os.path.exists(mixed_audio_path):
                try:
                    os.remove(mixed_audio_path)
                except Exception:
                    pass
            # Clean up wav if exists
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
                    
        except Exception as render_err:
            print(f"Render/Muxing failed: {render_err}")
            # Fallback to copy sample.mp4 in case of error
            src_video = os.path.join(STATIC_DIR, "sample.mp4")
            shutil.copy2(src_video, dest_video_path)
            
        update_db(85, "Rendering Video", "Rendering Video", "Video frames and audio track synchronized.")
        
        update_db(90, "Finalizing", "Finalizing", "Encoding output video to MP4 format...")
        time.sleep(sleep_step)
        update_db(95, "Finalizing", "Finalizing", "Optimizing for web playback (FastStart)...")
        
        # Bersihkan sub_progress dan tandai task selesai
        clear_sub_progress()
        update_db(100, "Completed", "Completed", "Video generation completed successfully!", status="Completed")
        
        title = f"Video: {actual_subject[:40]}..." if len(actual_subject) > 40 else actual_subject
        video_id = task_id
        download_url = f"/static/video_{task_id}.mp4"
        created_at_str = datetime.now().isoformat()
        
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO videos (id, title, prompt, duration, aspect_ratio, download_url, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (video_id, title, actual_subject, audio_duration, aspect_ratio, download_url, created_at_str)
            )
            cur.close()
            conn.close()
        except Exception as db_e:
            print(f"Failed to insert video record: {db_e}")
            raise db_e
    except Exception as err:
        import traceback
        traceback.print_exc()
        print(f"Error processing task {task_id} in background: {err}")
        logs.append(log_line("System Error", f"Task processing failed: {str(err)}"))
        # Bersihkan sub_progress saat task gagal
        clear_sub_progress()
        try:
            conn = get_db_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "UPDATE tasks SET status = 'Failed', step = 'Failed', logs = %s, sub_progress = NULL WHERE id = %s",
                (json.dumps(logs), task_id)
            )
            cur.close()
            conn.close()
        except Exception as db_e:
            print(f"Failed to update failure state in DB: {db_e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def get_task_status_db(task_id: str) -> Dict[str, Any]:
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
        task = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying task status: {e}")
        raise HTTPException(status_code=500, detail="Database query error")
        
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    logs_data = task["logs"]
    if isinstance(logs_data, str):
        logs_data = json.loads(logs_data)
    elif logs_data is None:
        logs_data = []
        
    # Parse sub_progress dari JSONB (bisa berupa dict atau string)
    sub_progress_data = task.get("sub_progress")
    if isinstance(sub_progress_data, str):
        try:
            sub_progress_data = json.loads(sub_progress_data)
        except Exception:
            sub_progress_data = None

    return {
        "task_id": task["id"],
        "video_subject": task["subject"],
        "status": task["status"],
        "progress": task["progress"],
        "step": task["step"],
        "logs": logs_data,
        "sub_progress": sub_progress_data,
        "video_url": f"/static/video_{task_id}.mp4" if task["status"] == "Completed" else None,
        "image_style": task.get("image_style") or "gtav",
        "character_prompt": task.get("character_prompt")
    }

@app.post("/api/v1/videos")
def create_video_task(req: VideoRequest):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    config = load_config()
    tts_provider = config.get("tts_provider", "edge-tts")
    if "Standard" in req.voice_name or "chatterbox" in req.voice_name:
        tts_provider = "local-chatterbox"
    
    duration_seconds = float(req.paragraph_number * 20.0)
    created_at = time.time()
    status = "processing"
    progress = 0
    step = "LLM Scripting"
    
    def log_line(category: str, message: str):
        log_time = datetime.now()
        return f"[{log_time.strftime('%Y-%m-%d %H:%M:%S')}] [{category}] {message}"
    
    logs = [log_line("LLM Scripting", "Initializing video generation task...")]
    
    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks (
                id, subject, script, aspect_ratio, voice_name, language, 
                paragraph_number, duration_seconds, created_at, status, progress, step, logs,
                local_steps, local_cfg, local_seed, local_negative_prompt, image_style, character_prompt
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                task_id, req.video_subject, "", req.video_aspect_ratio, req.voice_name, 
                req.language, req.paragraph_number, duration_seconds, created_at, 
                status, progress, step, json.dumps(logs),
                req.local_steps, req.local_cfg, req.local_seed, req.local_negative_prompt,
                req.image_style, req.character_prompt
            )
        )
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting task to DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize task in database")
        
    threading.Thread(
        target=process_task_background,
        args=(task_id, req.video_subject, req.video_aspect_ratio, req.voice_name, req.language, req.paragraph_number, duration_seconds, tts_provider),
        daemon=True
    ).start()
    
    return get_task_status_db(task_id)

@app.get("/api/v1/tasks")
def get_all_tasks():
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM tasks ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying tasks: {e}")
        return []
        
    tasks = []
    for row in rows:
        logs_data = row["logs"]
        if isinstance(logs_data, str):
            logs_data = json.loads(logs_data)
        elif logs_data is None:
            logs_data = []
            
        # Parse sub_progress dari JSONB
        sp = row.get("sub_progress")
        if isinstance(sp, str):
            try:
                sp = json.loads(sp)
            except Exception:
                sp = None

        tasks.append({
            "task_id": row["id"],
            "video_subject": row["subject"],
            "status": row["status"],
            "progress": row["progress"],
            "step": row["step"],
            "logs": logs_data,
            "sub_progress": sp,
            "video_url": f"/static/video_{row['id']}.mp4" if row["status"] == "Completed" else None
        })
    return tasks

@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    return get_task_status_db(task_id)

@app.post("/api/v1/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT status, logs FROM tasks WHERE id = %s", (task_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Task not found")
        
        current_status = row[0]
        logs_data = row[1]
        if isinstance(logs_data, str):
            logs_data = json.loads(logs_data)
        elif logs_data is None:
            logs_data = []

        if current_status.lower() in ["completed", "failed", "cancelled"]:
            cur.close()
            conn.close()
            return {"status": "ok", "message": f"Task is already {current_status}"}

        # Log the cancellation
        log_time = datetime.now()
        logs_data.append(f"[{log_time.strftime('%Y-%m-%d %H:%M:%S')}] [Cancellation] Task cancelled by user.")
        
        cur.execute(
            "UPDATE tasks SET status = 'cancelled', logs = %s WHERE id = %s",
            (json.dumps(logs_data), task_id)
        )
        cur.close()
        conn.close()
        return {"status": "ok", "message": "Task cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail="Database update error")

@app.delete("/api/v1/tasks/{task_id}")
def delete_task(task_id: str):
    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT id FROM tasks WHERE id = %s", (task_id,))
        if not cur.fetchone():
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Task not found")

        cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
        cur.execute("DELETE FROM videos WHERE id = %s", (task_id,))
        cur.close()
        conn.close()

        cleanup_task_files(task_id)
        return {"status": "ok", "message": "Task and associated files deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting task: {e}")
        raise HTTPException(status_code=500, detail="Database delete error")

@app.post("/api/v1/tasks/{task_id}/resume")
def resume_task(task_id: str):
    try:
        conn = get_db_conn()
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
        task = cur.fetchone()
        if not task:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Task not found")

        current_status = task["status"].lower() if task["status"] else ""
        if current_status not in ["cancelled", "failed"]:
            cur.close()
            conn.close()
            raise HTTPException(
                status_code=400, 
                detail=f"Only cancelled or failed tasks can be resumed. Current status: {task['status']}"
            )

        log_time = datetime.now()
        logs_data = task["logs"]
        if isinstance(logs_data, str):
            logs_data = json.loads(logs_data)
        elif logs_data is None:
            logs_data = []
        
        logs_data.append(f"[{log_time.strftime('%Y-%m-%d %H:%M:%S')}] [System] Resuming task... Progress reset to 0.")
        
        cur.execute(
            "UPDATE tasks SET status = 'processing', progress = 0, step = 'LLM Scripting', logs = %s WHERE id = %s",
            (json.dumps(logs_data), task_id)
        )
        cur.close()
        conn.close()

        cleanup_task_files(task_id)

        config = load_config()
        tts_provider = config.get("tts_provider", "edge-tts")
        if "Standard" in task["voice_name"] or "chatterbox" in task["voice_name"]:
            tts_provider = "local-chatterbox"

        threading.Thread(
            target=process_task_background,
            args=(
                task_id, 
                task["subject"], 
                task["aspect_ratio"], 
                task["voice_name"], 
                task["language"], 
                task["paragraph_number"], 
                task["duration_seconds"], 
                tts_provider
            ),
            daemon=True
        ).start()

        return {"status": "ok", "message": "Task resumed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error resuming task: {e}")
        raise HTTPException(status_code=500, detail="Database update error")

@app.get("/api/v1/videos")
def get_videos():
    try:
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying videos: {e}")
        return []
        
    videos = []
    for row in rows:
        videos.append({
            "id": row["id"],
            "title": row["title"],
            "prompt": row["prompt"],
            "duration": row["duration"],
            "aspect_ratio": row["aspect_ratio"],
            "download_url": row["download_url"],
            "created_at": row["created_at"]
        })
    return videos

@app.get("/api/v1/config")
def get_config():
    return load_config()

@app.post("/api/v1/config")
def update_config(config: Dict[str, Any]):
    save_config(config)
    return config

class ScriptGenerateRequest(BaseModel):
    subject: str
    language: str
    paragraphs: int

def generate_simulated_script(subject: str, language: str, paragraphs: int) -> str:
    sub_lower = subject.lower()
    theme = "general"
    if any(k in sub_lower for k in ["space", "galaxy", "planet", "star", "universe", "nasa", "astronomy"]):
        theme = "space"
    elif any(k in sub_lower for k in ["ocean", "sea", "fish", "marine", "shark", "water", "aquatic", "mariana"]):
        theme = "ocean"
    elif any(k in sub_lower for k in ["history", "war", "past", "ancient", "century", "emu", "empire"]):
        theme = "history"
    elif any(k in sub_lower for k in ["horor", "horror", "gunung", "mountain", "pendakian", "climb", "hike", "semeru", "kebaya", "kuning", "danau", "lake", "ranukumbolo", "kumbolo", "hantu", "setan", "mistis", "misteri"]):
        theme = "horror_climb"

    templates_en = {
        "space": [
            "Did you know that outer space is completely silent? Sound waves require a medium like air or water to travel, and because space is a vacuum, silence reigns supreme across the cosmos.",
            "Furthermore, our neighbor Venus has incredibly bizarre day-night cycles. A single day on Venus is longer than a whole year on Earth. It spins so slowly that it takes 243 Earth days to make one rotation, but only 225 Earth days to orbit the Sun.",
            "In addition, stars themselves are giant nuclear engines. The light we see from faraway stars has traveled for millions of light-years. When you look up at the night sky, you are literally looking back in time.",
            "Finally, the scale of the universe is almost impossible to comprehend. There are more stars in the observable universe than there are grains of sand on all the beaches of Earth combined, each potentially hosting its own planets."
        ],
        "ocean": [
            "The deep ocean remains one of the final frontiers of exploration on our planet. It is so vast and hostile that more humans have actually walked on the moon than have visited the deepest parts of our oceans.",
            "Deep down in the Mariana Trench, the water pressure is so intense it is equivalent to an elephant standing on your thumb. Yet, life somehow thrives in this extreme and dark environment.",
            "Creatures here have evolved unbelievable adaptations like bioluminescence. They generate their own light using chemical reactions, glowing in the pitch black to attract mates or capture prey.",
            "We have only mapped a small percentage of the ocean floor. Every deep-sea expedition discovers species that are entirely new to science, reminding us how little we know about our own world."
        ],
        "history": [
            "History is filled with events so strange they sound like fiction. In 1932, the Australian military literally declared war on emus to protect wheat crops in Western Australia.",
            "Armed with machine guns, soldiers were sent to combat the crop-destroying birds. However, the emus proved to be highly evasive and split into small groups, easily dodging the military's efforts.",
            "This became known as the Great Emu War. Despite multiple attempts, the emus effectively won, leaving the government to seek other methods of crop protection.",
            "It is a fascinating reminder of how human plans can go completely awry when facing the unexpected tactics of nature, and how history preserves these bizarre moments."
        ],
        "horror_climb": [
            "The air grew biting cold as we passed the last shelter on the slope of Mount Semeru. Whispering winds rushed through the ancient pine trees, carrying an eerie, distant hum.",
            "Suddenly, Andi turned to me and whispered with a trembling voice, \"I'm freezing, and I swear I hear whispering from the woods! Are we being followed?\"",
            "Hearing that, our guide Pak Joko immediately warned us, \"Do not look back! The mountain plays tricks on the mind. Focus on your steps and keep moving.\"",
            "I clutched my pack tightly and hurried forward, trying to ignore the chilling sensation of unseen eyes staring at us from the deep shadows."
        ],
        "general": [
            f"Let's explore the fascinating topic of {subject}. This subject has captivated minds for generations, presenting unique questions and sparking endless curiosity across various fields of study.",
            "When we look closer, we find a rich complexity that challenges our initial assumptions. Experts have dedicated lifetimes to understanding its intricacies, revealing new layers of knowledge.",
            "In today's fast-paced world, this topic is more relevant than ever. It shapes how we interact with technology, culture, and society, influencing future developments in unexpected ways.",
            "Ultimately, understanding this area allows us to broaden our horizons and appreciate the interconnectedness of our global community, prompting us to ask even deeper questions."
        ]
    }

    templates_es = {
        "space": [
            "¿Sabías que el espacio exterior es completamente silencioso? Las ondas de sonido requieren un medio como el aire para viajar, y al ser un vacío, el silencio reina en el cosmos.",
            "Además, Venus tiene un ciclo de día y noche increíblemente extraño. Un solo día en Venus es más largo que un año entero en la Tierra, tardando 243 días terrestres en rotar una vez.",
            "Las estrellas son gigantescos motores nucleares. La luz que vemos de las estrellas lejanas ha viajado durante millones de años luz, por lo que miramos al pasado al ver el cielo.",
            "La escala del universo es casi imposible de comprender. Hay más estrellas en el universo observable que granos de arena en todas las playas de la Tierra combinadas."
        ],
        "ocean": [
            "El océano profundo sigue siendo una de las últimas fronteras de exploración en nuestro planeta. Es tan vasto que más humanos han pisado la luna que visitado el fondo del mar.",
            "En la Fosa de las Marianas, la presión del agua es tan intensa como un elefante parado sobre tu pulgar, pero aun así, la vida prospera de manera sorprendente.",
            "Las criaturas han desarrollado adaptaciones increíbles como la bioluminiscencia. Generan su propia luz en la oscuridad absoluta para atraer parejas o capturar presas.",
            "Solo hemos mapeado un pequeño porcentaje del fondo marino. Cada expedición descubre especies completamente nuevas para la ciencia y la biología moderna."
        ],
        "history": [
            "La historia está llena de eventos tan extraños que parecen ficción. In 1932, el ejército australiano declaró la guerra a los emúes para proteger los cultivos de trigo.",
            "Armados con ametralladoras, los soldados intentaron combatir a las aves. Sin embargo, los emúes resultaron ser esquivos y se dispersaron en grupos pequeños.",
            "Este conflicto se conoció como la Gran Guerra del Emú. A pesar de los esfuerzos militares, los emúes ganaron de manera efectiva el enfrentamiento.",
            "Es un recordatorio fascinante de cómo los planes humanos pueden fallar ante la naturaleza, y cómo la historia conserva momentos tan peculiares."
        ],
        "horror_climb": [
            "El aire se volvió helado cuando pasamos el último refugio en la ladera del Monte Semeru. Vientos susurrantes soplaban entre los viejos pinos, trayendo un zumbido misterioso y lejano. El guía nos advirtió que no miráramos atrás, pero la sensación de ser observados era imposible de ignorar.",
            "Al acercarse la medianoche, una niebla espesa y congelante nos rodeó, apagando nuestras linternas y aislándonos en el frío. De repente, pasos suaves nos seguían por detrás a nuestro mismo ritmo, pero al detenernos, los pasos se desvanecían en el silencio.",
            "El pánico comenzó cuando las brújulas giraron sin control y el camino parecía regresar al mismo roble nudoso. Rezando en silencio, avanzamos con todas nuestras fuerzas, ignorando las voces que nos llamaban. Con la luz del amanecer, las sombras se retiraron y nos salvaron del horror del monte."
        ],
        "general": [
            f"Exploremos el fascinante tema de {subject}. Este asunto ha cautivado mentes durante generaciones, presentando preguntas únicas y despertando curiosidad en diversos campos.",
            "Al mirar más de cerca, encontramos una rica complejidad que desafía nuestras suposiciones iniciales. Los expertos dedican vidas enteras a comprender sus detalles.",
            "En el mundo acelerado de hoy, este tema es más relevante que nunca. Modela cómo interactuamos con la tecnología y la cultura contemporánea.",
            "Comprender esta área nos permite ampliar nuestros horizontes y apreciar la interconexión global, impulsándonos a hacer preguntas aún más profundas."
        ]
    }

    templates_fr = {
        "space": [
            "Saviez-vous que l'espace est complètement silencieux ? Les ondes sonores ont besoin d'un milieu comme l'air pour voyager, et comme l'espace est un vide, le silence y règne.",
            "De plus, Vénus a un cycle jour-nuit incroyablement bizarre. Une seule journée sur Vénus est plus longue qu'une année entière sur Terre, prenant 243 jours terrestres.",
            "Les étoiles sont de gigantesques moteurs nucléaires. La lumière que nous voyons des étoiles lointaines a voyagé pendant des millions d'années-lumière.",
            "L'échelle de l'univers est presque impossible à comprendre. Il y a plus d'étoiles dans l'univers observable que de grains de sable sur toutes les plages de la Terre."
        ],
        "ocean": [
            "L'océan profond reste l'une des dernières frontières de l'exploration sur notre planète. Plus d'humains ont marché sur la lune que visité les abysses.",
            "Dans la fosse des Mariannes, la pression de l'eau est équivalente à un éléphant debout sur votre pouce. Pourtant, la vie y trouve son chemin.",
            "Les créatures y ont développé des adaptations incroyables comme la bioluminescence. Elles produisent leur propre lumière pour chasser ou s'accoupler.",
            "Nous n'avons cartographié qu'un faible pourcentage des fonds marins. Chaque expédition révèle de nouvelles espèces totalement inconnues de la science."
        ],
        "history": [
            "L'histoire est pleine d'événements si étranges qu'ils semblent fictifs. En 1932, l'armée australienne a déclaré la guerre aux émeus pour protéger les récoltes.",
            "Armés de mitrailleuses, les soldats ont tenté de combattre ces oiseaux. Cependant, les émeus se sont révélés très esquiveurs et tactiques.",
            "Cet événement est resté célèbre sous le nom de la Grande Guerre des Émeus. Contre toute attente, les émeus ont remporté le conflit.",
            "C'est un rappel fascinant de la manière dont les plans humains peuvent échouer face à la nature, et comment l'histoire conserve ces moments insolites."
        ],
        "horror_climb": [
            "L'air devint glacial alors que nous dépassions le dernier abri sur le mont Semeru. Les vents chuchotaient dans les pins, apportant un bourdonnement étrange. Le guide nous interdit de nous retourner, mais la sensation d'être épiés par des yeux invisibles dans l'obscurité était terrifiante.",
            "À l'approche de minuit, un brouillard givrant nous enveloppa, bloquant nos lampes et nous isolant dans un vide gris. Soudain, des pas légers imitèrent les nôtres. Quand nous nous arrêtions, ces bruits de pas flottaient un instant avant de s'évanouir.",
            "La panique monta quand nos boussoles s'affolèrent. Rassemblant nos forces, nous avons continué en priant, ignorant les voix qui nous appelaient. À l'aube, les ombres disparurent, nous laissant épuisés mais vivants après cette nuit d'épouvante."
        ],
        "general": [
            f"Explorons le sujet fascinant de {subject}. Cette thématique captive les esprits depuis des générations, suscitant des questions uniques et une grande curiosité.",
            "En y regardant de plus près, nous découvrons une complexité qui remet en question nos certitudes. Les experts y consacrent des vies entières.",
            "Dans notre monde moderne, ce sujet est plus pertinent que jamais. Il influence notre rapport à la technologie et à la société actuelle.",
            "Comprendre ce domaine nous permet d'élargir nos horizons et d'apprécier les liens qui nous unissent, nous incitant à aller encore plus loin."
        ]
    }

    templates_de = {
        "space": [
            "Wussten Sie, dass das Weltall völlig geräuschlos ist? Schallwellen benötigen ein Medium wie Luft, und da das All ein Vakuum ist, herrscht dort absolute Stille.",
            "Zudem hat die Venus einen äußerst seltsamen Tag-Nacht-Rhythmus. Ein einziger Tag auf der Venus dauert länger als ein ganzes Jahr auf der Erde.",
            "Sterne sind gigantische Kernreaktoren. Das Licht, das wir von fernen Sternen sehen, war Millionen von Lichtjahren unterwegs, bevor es uns erreicht.",
            "Die Dimensionen des Universums sind unvorstellbar. Es gibt im beobachtbaren Universum mehr Sterne als Sandkörner an allen Stränden der Erde zusammen."
        ],
        "ocean": [
            "Die Tiefsee ist eine der letzten unberührten Grenzen unseres Planeten. Mehr Menschen waren auf dem Mond als an den tiefsten Punkten der Ozeane.",
            "Im Marianengraben ist der Wasserdruck so extrem wie ein Elefant, der auf Ihrem Daumen steht. Dennoch gedeiht dort erstaunliches Leben.",
            "Die dortigen Lebewesen nutzen biolumineszierende Reaktionen. Sie erzeugen ihr eigenes Licht, um in der Dunkelheit Beute oder Partner anzulocken.",
            "Bisher ist nur ein Bruchteil des Meeresbodens kartiert. Jede Expedition bringt neue, der Wissenschaft völlig unbekannte Arten ans Licht."
        ],
        "history": [
            "Die Geschichte ist voller skurriler Ereignisse. Im Jahr 1932 erklärte das australische Militär den Emus den Krieg, um die Weizenernte zu schützen.",
            "Mit Maschinengewehren bewaffnete Soldaten wurden ausgesandt. Doch die Emus erwiesen sich als flink und wichen den Truppen geschickt aus.",
            "Dieses Ereignis ging als der Große Emu-Krieg in die Geschichte ein. Trotz des Einsatzes von Waffen gewannen die Emus den Konflikt.",
            "Dies zeigt auf humorvolle Weise, wie menschliche Pläne an den unvorhersehbaren Taktiken der Natur scheitern können."
        ],
        "horror_climb": [
            "Die Luft wurde eisig, als wir die letzte Schutzhütte am Hang des Mount Semeru passierten. Winselnde Winde wehten durch die alten Kiefern und trugen ein unheimliches Summen mit sich. Der Führer warnte uns, nicht zurückzublicken, aber das Gefühl, beobachtet zu werden, war übermächtig.",
            "Gegen Mitternacht zog dichter Nebel auf, der das Licht unserer Taschenlampen verschluckte und uns isolierte. Plötzlich waren leise Schritte hinter uns zu hören, die genau unserem Tempo entsprachen. Wenn wir anhielten, verstummten sie kurz darauf ebenfalls.",
            "Panik brach aus, als unsere Kompasse verrücktspielten und der Pfad im Kreis zu führen schien. Mit letzter Kraft gingen wir betend vorwärts und ignorierten die Stimmen, die uns riefen. Erst im Morgengrauen wichen die Schatten und ließen uns erschöpft, aber lebend zurück."
        ],
        "general": [
            f"Lassen Sie uns das faszinierende Thema {subject} näher beleuchten. Dieses Thema fasziniert die Menschen seit Generationen und wirft spannende Fragen auf.",
            "Bei genauerer Betrachtung offenbart sich eine Komplexität, die unsere Annahmen herausfordert. Experten widmen ihr Leben der Erforschung dieser Details.",
            "In der heutigen Zeit ist diese Thematik relevanter denn je. Sie beeinflusst unseren Umgang mit Technologie, Kultur und der Gesellschaft.",
            "Ein tiefes Verständnis dieses Bereichs erweitert unseren Horizont und lässt uns die Zusammenhänge in einer globalisierten Welt besser verstehen."
        ]
    }

    templates_zh = {
        "space": [
            "你知道太空是完全寂静的吗？声波需要空气等介质传播，而在真空的太空中，寂静无声是宇宙的常态。",
            "此外，金星的昼夜交替非常奇特。金星上的一天比地球上的一年还要长。它自转极慢，需要243个地球日，而绕太阳公转只需225天。",
            "星星本身就是巨大的核能引擎。我们看到的星光已经旅行了数百万光年。当你仰望星空时，你实际上是在凝望历史。",
            "宇宙的尺度之大令人难以置信。可观测宇宙中的恒星数量比地球上所有沙滩上的沙粒总和还要多。"
        ],
        "ocean": [
            "深海是地球上最后的探索边界之一。这片领域如此辽阔，登上月球的人类甚至比造访海洋最深处的人还要多。",
            "在马里亚纳海沟深处，水压相当于一只大象站在你的大拇指上。然而，生命依然以奇特的方式在黑暗中繁衍。",
            "这里的生物进化出了令人惊叹的生物发光特征。它们在漆黑的深海中通过化学反应产生光亮，以吸引配偶或诱捕猎物。",
            "我们只绘制了极小部分的海洋地图。每一次深海科考都会发现全新物种，提醒着我们对地球的了解是多么有限。"
        ],
        "history": [
            "历史上充斥着匪夷所思的真实事件。1932年，澳大利亚军队曾向鸸鹋宣战，以保护西澳大利亚州的麦田免受破坏。",
            "士兵们配备了机关枪去对付这些巨大的鸟类。然而，鸸鹋机警异常，分散成小群轻松避开了军队的围剿。",
            "这被称为大鸸鹋战争。尽管军方付出了努力，鸸鹋最终还是赢得了这场胜利，政府不得不寻找其他保护作物的方法。",
            "这是一个有趣的提醒，表明人类面对自然规律和奇特动物时，计划可能会彻底落空，也为历史留下了这段奇特一幕。"
        ],
        "horror_climb": [
            "当我们穿过塞梅鲁火山山坡上的最后一个避难所时，空气变得刺骨般寒冷。风穿过古老的松树，带来源源不断的诡异声响。向导警告我们不要回头，但那种在黑暗中被监视的感觉是如此真实，令人无法忽视。",
            "临近午夜，冰冷的浓雾笼罩了我们，吞噬了手电筒的光芒。突然，身后传来了轻轻的脚步声，与我们的步伐保持着一致的节奏。但每当我们停下时，那脚步声便也跟着消失在寂静中。",
            "恐慌在蔓延，我们的指南针疯狂旋转，眼前的山路似乎一直在把我们带回同一个死树旁。我们默默祈祷并用尽全力向前走，无视那些从黑暗深渊中呼喊我们名字的声音。当黎明的第一缕光线穿透山顶，阴影终于退去，我们得以生还。"
        ],
        "general": [
            f"让我们共同探讨关于{subject}的引人入胜的话题。这一领域在漫长的岁月里持续吸引着人们的目光，激发着跨领域的深入思考与探索。",
            "深入观察后，我们会发现其背后隐藏的复杂关联，它往往会颠覆我们最初的认知，这也是许多学者终其一生研究的魅力所在。",
            "在当今快速发展的世界中，这一主题的现实意义比以往任何时候都更加重大，它影响着我们与科技、文化和社会的良性互动。",
            "最终，对这一课题的探索有助于拓展我们的认知疆界，使我们更深刻地理解全球化背景下各种社会现象的内在联系。"
        ]
    }

    templates_id = {
        "space": [
            "Tahukah Anda bahwa luar angkasa benar-benar sunyi? Gelombang suara membutuhkan media seperti udara atau air untuk merambat, dan karena luar angkasa adalah ruang hampa, keheningan total menguasai seluruh kosmos.",
            "Selain itu, tetangga kita Venus memiliki siklus siang-malam yang sangat aneh. Satu hari di Venus lebih lama daripada satu tahun penuh di Bumi. Venus berputar sangat lambat sehingga membutuhkan 243 hari Bumi untuk sekali rotasi.",
            "Bintang-bintang itu sendiri adalah mesin nuklir raksasa. Cahaya yang kita lihat dari bintang-bintang yang sangat jauh telah melakukan perjalanan selama jutaan tahun cahaya, sehingga Anda secara harfiah melihat masa lalu.",
            "Skala alam semesta ini hampir tidak mungkin dipahami secara akal. Ada lebih banyak bintang di alam semesta yang teramati daripada butiran pasir di seluruh pantai di planet Bumi jika digabungkan."
        ],
        "ocean": [
            "Laut dalam tetap menjadi salah satu perbatasan eksplorasi terakhir di planet kita. Wilayah ini sangat luas dan ekstrem sehingga lebih banyak manusia yang telah berjalan di bulan daripada mengunjungi dasar samudra terdalam.",
            "Di kedalaman Palung Mariana, tekanan air sangat hebat sehingga setara dengan seekor gajah yang berdiri di atas jempol Anda, namun kehidupan entah bagaimana tetap tumbuh subur.",
            "Makhluk-makhluk di sini telah mengembangkan adaptasi luar biasa seperti bioluminesensi. Mereka menghasilkan cahaya sendiri dalam kegelapan pekat untuk menarik pasangan atau berburu.",
            "Kita baru memetakan sebagian kecil dari dasar laut. Setiap ekspedisi laut dalam selalu menemukan spesies baru, mengingatkan kita betapa sedikitnya yang kita ketahui tentang dunia kita sendiri."
        ],
        "history": [
            "Sejarah dipenuhi dengan peristiwa unik yang terdengar seperti fiksi. Pada tahun 1932, militer Australia secara harfiah menyatakan perang terhadap burung emu untuk melindungi tanaman gandum.",
            "Dipersenjatai dengan senapan mesin, tentara dikirim untuk membasmi burung-burung tersebut. Namun, burung emu terbukti sangat gesit dan menyebar menjadi kelompok kecil, dengan mudah menghindari tembakan.",
            "Peristiwa ini dikenal sebagai Perang Emu Besar. Terlepas dari upaya militer, burung emu secara efektif memenangkan konflik tersebut, memaksa pemerintah mencari metode lain.",
            "Ini adalah pengingat menarik tentang bagaimana rencana manusia dapat gagal total saat menghadapi taktik alam yang tak terduga, dan bagaimana sejarah menyimpan momen aneh ini."
        ],
        "horror_climb": [
            "Sumpah, dinginnya Gunung Semeru malam itu bener-bener menusuk tulang pas kita ngelewatin pos terakhir. Angin malam berhembus di sela pohon pinus, suaranya kayak bisikan gaib yang bikin merinding parah.",
            "Tiba-tiba, si Andi nyenggol bahu gue sambil bisik-bisik ketakutan, \"Gokil, gue merinding parah! Rasanya kayak ada sepasang mata merah yang ngeliatin kita terus dari balik semak-semak.\"",
            "Mendengar itu, Pak Joko pemandu kita langsung ngasih peringatan keras, \"Jangan nengok ke belakang! Tetap fokus sama jalan di depan dan jalan terus. Kosongin pikiran kalian.\"",
            "Gue pun berusaha jalan lebih cepet sambil nahan takut setengah mati, apalagi pas ngeliat sesosok bayangan putih tanpa wajah berdiri diem di kejauhan."
        ],
        "general": [
            f"Mari kita telusuri topik menarik mengenai {subject}. Subjek ini telah memikat pikiran banyak generasi, menghadirkan pertanyaan unik dan memicu rasa ingin tahu yang tak ada habisnya.",
            "Ketika kita melihat lebih dekat, kita menemukan kompleksitas kaya yang menantang asumsi awal kita. Para ahli telah mendedikasikan hidup mereka untuk memahaminya.",
            "Di dunia modern yang serba cepat saat ini, topik ini menjadi lebih relevan dari sebelumnya. Ini membentuk cara kita berinteraksi dengan teknologi, budaya, dan masyarakat.",
            "Pada akhirnya, memahami bidang ini memungkinkan kita memperluas wawasan dan menghargai keterkaitan komunitas global kita, mendorong kita untuk mengajukan pertanyaan yang lebih dalam."
        ]
    }

    templates = templates_en
    if language == "es":
        templates = templates_es
    elif language == "fr":
        templates = templates_fr
    elif language == "de":
        templates = templates_de
    elif language == "zh":
        templates = templates_zh
    elif language == "id":
        templates = templates_id

    theme_list = templates.get(theme, templates["general"])
    result_paras = []
    for i in range(paragraphs):
        para_template = theme_list[i % len(theme_list)]
        result_paras.append(para_template)

    return "\n\n".join(result_paras)

def get_script_system_prompt(paragraphs: int, language: str) -> str:
    # Mengembalikan system prompt untuk pembuatan skrip.
    # Secara otomatis mendeteksi genre dan nada cerita berdasarkan subjek yang diberikan pengguna.
    # Menghindari bias horor dan tidak menyertakan contoh dialog horor seperti "Aku merinding...".
    # Menggunakan POV orang pertama (1st person POV).
    # Bahasa Indonesia menggunakan gaya kasual/gaul ramah Gen Z agar terasa seperti vlog nyata.
    return (
        f"You are an elite, professional creative storyteller and script writer. "
        f"Your goal is to write a highly engaging, immersive, and vivid spoken script/story about the user's topic. "
        f"You must automatically detect the genre and tone of the story from the user's provided subject "
        f"(e.g., sci-fi/adventure for space, horror for spooky/creepy, educational/documentary for facts/nature, drama/slice-of-life for personal stories). "
        f"Adapt your storytelling style and tone to match the detected genre without forcing any pre-determined genre. "
        f"The story/script MUST be written from a first-person point of view (1st person POV) as a personal experience. "
        f"For English, write it in a casual, conversational storytelling tone (using natural pauses like '...' and bracketed paralinguistic tags like [sigh], [gasp], [laughter], [chuckle]). "
        f"For Indonesian, write it in a casual, slightly slangy Gen Z-friendly language (bahasa santai/gaul, e.g., using 'gue/lo/kita' instead of 'saya/anda/kami', and using common informal terms like 'bener-bener', 'parah', 'gokil' to make it feel like a real vlog storytelling of the detected genre). "
        f"You must write it as a natural narrative with no speaker names or character prefixes (do NOT write 'Saya:', 'Andi:', 'Narator:', etc.). "
        f"Instead, embed character dialogue directly within the paragraphs using quotation marks (e.g., when a character speaks, use quotation marks like \"Aku rasa...\" or \"Kok bisa ya...\"). "
        f"Do not write generic marketing or summary content. It should have exactly {paragraphs} paragraphs, written in {language} language. "
        f"Write ONLY the script story text, with no introduction, no outro, and no stage directions/audio cues."
    )

def get_humanizer_system_prompt(paragraphs: int) -> str:
    # Mengembalikan system prompt untuk humanizer agent.
    # Memperhalus skrip agar terdengar alami, percakapan, dan mentah tanpa contoh dialog horor yang di-hardcode.
    return (
        f"You are an expert Humanizer Agent. Your job is to take a draft storytelling script and refine it to make it sound incredibly natural, conversational, and raw as if spoken by a real person in a vlog or podcast.\n"
        f"For English, rewrite the script in a casual, conversational storytelling tone (using natural pauses like '...' and inserting paralinguistic tag tokens in brackets like [sigh], [gasp], [laughter], [chuckle] where a speaker would naturally express emotion).\n"
        f"For Indonesian, rewrite the script in a casual, slightly slangy Gen Z-friendly language (bahasa santai/gaul, e.g., using 'gue/lo/kita' instead of 'saya/anda/kami', and using common informal terms like 'bener-bener', 'parah', 'gokil', 'deh', 'sih', 'kok', and inserting paralinguistic tag tokens in brackets like [sigh], [gasp], [laughter], [chuckle]).\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"1. The story/script MUST remain in first-person POV (1st person POV) as a personal experience.\n"
        f"2. You must NOT add any speaker names or character prefixes (do NOT write 'Saya:', 'Andi:', 'Narator:', etc.).\n"
        f"3. All character spoken dialogue MUST be enclosed in double quotes (e.g. \"Aku rasa...\" or \"Kok bisa ya...\") and narration MUST NOT have quotes.\n"
        f"4. Output ONLY the final humanized script text, maintaining exactly {paragraphs} paragraphs, with no extra introductions or outro notes."
    )

def generate_scene_image_prompt(scene_text: str, subject: str, image_style: str = None, character_prompt: str = None) -> str:
    # Membersihkan tag paralinguistik dalam tanda kurung siku (seperti [sigh], [gasp]).
    # Memisahkan narasi dari dialog, memprioritaskan narasi untuk mengekstrak konteks visual.
    # Mengekstrak kata kunci visual sambil memfilter kata pengisi/slang percakapan (seperti gue, lo, bener-bener, dll).
    # Menggabungkan image_style dan character_prompt ke dalam prompt deskriptif akhir.
    import re
    
    # 1. Bersihkan tag paralinguistik di dalam kurung siku
    cleaned_text = re.sub(r'\[[^\]]*\]', '', scene_text)
    
    # 2. Pisahkan narasi dan dialog. Dialog berada di dalam tanda kutip ganda.
    parts = re.split(r'("[^"]*")', cleaned_text)
    narration_parts = []
    dialogue_parts = []
    
    for part in parts:
        part_strip = part.strip()
        if not part_strip:
            continue
        if part_strip.startswith('"') and part_strip.endswith('"'):
            dialogue_parts.append(part_strip[1:-1].strip())
        else:
            narration_parts.append(part_strip)
            
    # Prioritaskan narasi untuk ekstraksi konteks visual
    visual_base = " ".join(narration_parts).strip()
    if not visual_base:
        # Jika tidak ada narasi, gunakan teks dialog
        visual_base = " ".join(dialogue_parts).strip()
    if not visual_base:
        visual_base = cleaned_text.strip()
        
    # 3. Filter kata-kata slang dan pengisi percakapan (case-insensitive)
    filler_words = {"gue", "lo", "bener-bener", "parah", "gokil", "sih", "deh", "kok"}
    filler_pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    
    cleaned_visual = re.sub(filler_pattern, '', visual_base, flags=re.IGNORECASE)
    
    # Bersihkan spasi ganda dan tanda baca gantung
    cleaned_visual = re.sub(r'\s+', ' ', cleaned_visual).strip()
    cleaned_visual = re.sub(r'\s*,\s*', ', ', cleaned_visual)
    cleaned_visual = re.sub(r'^\s*,\s*|\s*,\s*$', '', cleaned_visual)
    cleaned_visual = re.sub(r'\s*\.\s*', '. ', cleaned_visual)
    cleaned_visual = re.sub(r'^\s*\.\s*|\s*\.\s*$', '', cleaned_visual)
    cleaned_visual = cleaned_visual.strip()
    
    if not cleaned_visual:
        cleaned_visual = subject
        
    # 4. Gabungkan character_prompt, visual adegan yang dibersihkan, subjek, dan style
    prompt_parts = []
    if character_prompt and character_prompt.strip():
        prompt_parts.append(character_prompt.strip())
        
    prompt_parts.append(cleaned_visual)
    
    if subject and subject.strip():
        prompt_parts.append(f"{subject.strip()} setting")
        
    if image_style and image_style.strip():
        prompt_parts.append(f"{image_style.strip()} style")
        
    final_prompt = ", ".join(prompt_parts)
    # Bersihkan koma berturut-turut atau spasi berlebih
    final_prompt = re.sub(r',\s*,', ',', final_prompt)
    final_prompt = re.sub(r'\s+', ' ', final_prompt).strip()
    
    return final_prompt

def humanize_script(script: str, provider: str, api_key: str, language: str, paragraphs: int) -> str:
    print(f"Running Stage 2: Humanizer Agent pass using provider: {provider}")
    
    system_prompt = get_humanizer_system_prompt(paragraphs)
    
    try:
        if provider == "openai":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Draft Script to Humanize:\n{script}"}
                ]
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [
                        {"text": f"{system_prompt}\n\nDraft Script to Humanize:\n{script}"}
                    ]
                }]
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            
        elif provider == "groq":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Draft Script to Humanize:\n{script}"}
                ]
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        elif provider == "deepseek":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Draft Script to Humanize:\n{script}"}
                ]
            }
            response = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        print(f"Humanizer Agent failed: {e}. Returning raw draft script.")
        return script

@app.post("/api/v1/script/generate")
def generate_script_endpoint(req: ScriptGenerateRequest):
    config = load_config()
    provider = config.get("llm_provider", "openai")
    api_key = config.get(f"{provider}_api_key", "").strip()

    subject = req.subject
    language = req.language
    paragraphs = req.paragraphs

    if not api_key:
        print("No API key configured. Generating simulated script.")
        script = generate_simulated_script(subject, language, paragraphs)
        return {"script": script}

    print(f"Calling real LLM provider: {provider}")
    try:
        if provider == "openai":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": get_script_system_prompt(paragraphs, language)},
                    {"role": "user", "content": f"Subject: {subject}"}
                ]
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            script = response.json()["choices"][0]["message"]["content"].strip()

        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            prompt_text = f"{get_script_system_prompt(paragraphs, language)}\n\nSubject: {subject}"
            payload = {
                "contents": [{
                    "parts": [{"text": prompt_text}]
                }]
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            script = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        elif provider == "groq":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": get_script_system_prompt(paragraphs, language)},
                    {"role": "user", "content": f"Subject: {subject}"}
                ]
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            script = response.json()["choices"][0]["message"]["content"].strip()

        elif provider == "deepseek":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": get_script_system_prompt(paragraphs, language)},
                    {"role": "user", "content": f"Subject: {subject}"}
                ]
            }
            response = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            script = response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Unsupported provider {provider}. Generating simulated script.")
            script = generate_simulated_script(subject, language, paragraphs)
            return {"script": script}

        # Run Stage 2: Humanizer Agent Pass
        script = humanize_script(script, provider, api_key, language, paragraphs)
        return {"script": script}
    except Exception as e:
        print(f"Error generating script using {provider}: {e}. Falling back to simulated script.")
        script = generate_simulated_script(subject, language, paragraphs)
        return {"script": script}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

