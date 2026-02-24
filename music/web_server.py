#!/usr/bin/env python3
"""
FastAPI Music Server with Web Player
Provides REST API for music streaming, search, playlists, and AI image generation
"""

import asyncio
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import shutil
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from openai import OpenAI
from PIL import Image
import io

# Configuration
PORT = 9100
SERVER_IP = ""
MUSIC_DIR = Path(__file__).parent / "data"
IMAGE_CACHE_DIR = MUSIC_DIR / "cache" / "images"
PLAYLISTS_DIR = MUSIC_DIR / "playlists"
PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI AI Search Configuration
OPENAI_API_BASE = "http://192.168.13.228:8000/v1/"
OPENAI_API_KEY = "123"
OPENAI_MODEL = "Qwen3-30B-A3B"

# SiliconFlow Image Generation Configuration
SILICONFLOW_API_KEY = "sk-qnshhbcisgilevxquheykzdblutjagkjyfrodbntldipcbvg"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/images/generations"

# Initialize clients
ai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# Global mappings
_ID_MAPPING = {}
_REVERSE_MAPPING = {}
_IMAGE_ID_MAPPING = {}
_IMAGE_REVERSE_MAPPING = {}

# Active WebSocket connections
active_connections: List[WebSocket] = []


# Pydantic models
class SongInfo(BaseModel):
    id: int
    name: str
    path: str
    size: int
    image: Optional[str] = None
    duration: Optional[float] = None


class SearchResponse(BaseModel):
    total: int
    files: List[SongInfo]


class PlaylistCreate(BaseModel):
    name: str
    description: Optional[str] = None


class PlaylistUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    song_ids: Optional[List[int]] = None


class PlaylistItem(BaseModel):
    song_id: int
    added_at: str


class ImageGenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "Qwen/Qwen-Image-Edit-2509"
    negative_prompt: Optional[str] = ""
    image_size: Optional[str] = "1024x1024"
    batch_size: Optional[int] = 1
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5


class PlayHistory(BaseModel):
    song_id: int
    played_at: str


# Utility functions
def get_local_ip():
    """Get local IP address automatically"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def clean_song_name(filename):
    """Remove numeric prefix and .mp3 suffix from filename"""
    name = re.sub(r"^\d+\.\s*", "", filename)
    name = name.replace(".mp3", "")
    return name


def normalize_text(text):
    """Normalize text for better search matching"""
    text = re.sub(r"[\/《》\[\]()\-·\.]+", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]+", "", text)
    text = text.replace("周杰伦", "")
    text = text.replace("的", "")
    text = text.replace("、", "")
    return text.strip().lower()


def is_match(search_query, file_path):
    """Check if search query matches file path using intelligent matching"""
    search_lower = search_query.lower()

    if search_lower in file_path.lower():
        return True

    normalized_query = normalize_text(search_lower)
    normalized_path = normalize_text(file_path)

    if not normalized_query:
        return True

    if normalized_query in normalized_path:
        return True

    keywords = re.split(r"[\s\-·、]+", normalized_query)
    keywords = [k for k in keywords if k]

    if not keywords:
        return True

    for keyword in keywords:
        if keyword not in normalized_path:
            return False

    return True


def ai_search(search_query):
    """Use AI model to find matching songs"""
    if not _ID_MAPPING:
        return []

    try:
        songs_list = []
        max_songs = min(500, len(_ID_MAPPING))

        for idx, rel_path in list(_ID_MAPPING.items())[:max_songs]:
            file_path = MUSIC_DIR / rel_path
            songs_list.append({
                "id": idx,
                "name": clean_song_name(file_path.name),
                "path": rel_path
            })

        prompt = f"""你是一个音乐搜索助手。用户想找一首歌，请从提供的歌曲列表中找出最匹配的歌曲。

用户搜索词: "{search_query}"

歌曲列表（JSON格式）:
{json.dumps(songs_list, ensure_ascii=False, indent=2)}

要求：
1. 找出与用户搜索意图最匹配的歌曲（支持模糊匹配、语义理解、别名识别）
2. 如果有多首歌曲匹配，返回所有相关的歌曲，按相关度排序
3. 如果没有找到匹配的歌曲，返回空数组
4. 只返回歌曲ID列表，不要包含其他内容

请直接返回JSON数组，格式：[id1, id2, id3]"""

        request_data = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的音乐搜索助手，擅长理解用户的搜索意图并找出匹配的歌曲。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        }

        response = ai_client.chat.completions.create(**request_data)
        ai_response = response.choices[0].message.content.strip()

        import ast
        ai_response = ai_response.strip()
        if ai_response.startswith("[") and ai_response.endswith("]"):
            matched_ids = ast.literal_eval(ai_response)
            if isinstance(matched_ids, list):
                return [
                    int(i)
                    for i in matched_ids
                    if isinstance(i, int) and i in _ID_MAPPING
                ]

        return []

    except Exception as e:
        print(f"[DEBUG] AI search failed: {type(e).__name__}: {e}")
        return []


def get_image_url(mp3_path):
    """Get short image URL for an MP3 file if exists"""
    try:
        mp3_path = MUSIC_DIR / mp3_path
        png_path = mp3_path.with_suffix(".png")

        if png_path.exists():
            relative_path = str(png_path.relative_to(MUSIC_DIR))
            image_id = _IMAGE_REVERSE_MAPPING.get(relative_path)
            if image_id:
                server_ip = SERVER_IP if SERVER_IP else get_local_ip()
                return f"http://{SERVER_IP or get_local_ip()}:{PORT}/api/image/{image_id}"
        return None
    except Exception:
        return None


def build_id_mapping():
    """Build ID to filename mapping from MP3 files and PNG images"""
    global _ID_MAPPING, _REVERSE_MAPPING, _IMAGE_ID_MAPPING, _IMAGE_REVERSE_MAPPING

    mapping = {}
    idx = 1
    for file_path in sorted(MUSIC_DIR.glob("**/*.mp3")):
        if (
            "backup_original" in file_path.parts
            or ".tmp" in str(file_path)
            or "_temp" in str(file_path)
            or "/.venv/" in str(file_path)
            or "/__pycache__/" in str(file_path)
            or "/.pytest_cache/" in str(file_path)
            or ".git" in str(file_path).lower()
        ):
            continue
        mapping[idx] = str(file_path.relative_to(MUSIC_DIR))
        idx += 1

    _ID_MAPPING = mapping
    _REVERSE_MAPPING = {v: k for k, v in mapping.items()}

    # Build image ID mapping
    image_mapping = {}
    img_idx = 1
    for file_path in sorted(MUSIC_DIR.glob("**/*.png")):
        if (
            "backup_original" in file_path.parts
            or ".tmp" in str(file_path)
            or "_temp" in str(file_path)
            or "/.venv/" in str(file_path)
            or "/__pycache__/" in str(file_path)
            or "/.pytest_cache/" in str(file_path)
            or ".git" in str(file_path).lower()
            or "assets" in str(file_path).lower()
            or "cache" in str(file_path).lower()
        ):
            continue
        image_mapping[img_idx] = str(file_path.relative_to(MUSIC_DIR))
        img_idx += 1

    _IMAGE_ID_MAPPING = image_mapping
    _IMAGE_REVERSE_MAPPING = {v: k for k, v in image_mapping.items()}


# FastAPI app
app = FastAPI(title="Music Server API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "total_songs": len(_ID_MAPPING),
        "total_images": len(_IMAGE_ID_MAPPING)
    }


@app.get("/api/list", response_model=SearchResponse)
async def list_songs(q: str = Query("", description="Search query")):
    """List all songs with optional filtering"""
    mp3_files = []

    for idx, rel_path in _ID_MAPPING.items():
        file_path = MUSIC_DIR / rel_path

        if q and q.lower() not in rel_path.lower():
            continue

        info = SongInfo(
            id=idx,
            name=clean_song_name(file_path.name),
            path=rel_path,
            size=file_path.stat().st_size,
            image=get_image_url(rel_path)
        )
        mp3_files.append(info)

    return SearchResponse(total=len(mp3_files), files=mp3_files)


@app.get("/api/search", response_model=SearchResponse)
async def search_songs(q: str = Query(..., description="Search query")):
    """Search songs using hybrid (traditional + AI) search"""
    if not q:
        raise HTTPException(status_code=400, detail="Search query 'q' is required")

    mp3_files = []

    # Step 1: Traditional search
    traditional_matches = []
    for idx, rel_path in _ID_MAPPING.items():
        if is_match(q, rel_path):
            file_path = MUSIC_DIR / rel_path
            traditional_matches.append(SongInfo(
                id=idx,
                name=clean_song_name(file_path.name),
                path=rel_path,
                size=file_path.stat().st_size,
                image=get_image_url(rel_path)
            ))

    # Step 2: Decision based on results
    if len(traditional_matches) == 1:
        mp3_files = traditional_matches
    else:
        # Use AI search
        matched_ids = ai_search(q)

        if matched_ids:
            for idx in matched_ids:
                rel_path = _ID_MAPPING[idx]
                file_path = MUSIC_DIR / rel_path
                mp3_files.append(SongInfo(
                    id=idx,
                    name=clean_song_name(file_path.name),
                    path=rel_path,
                    size=file_path.stat().st_size,
                    image=get_image_url(rel_path)
                ))
        else:
            mp3_files = traditional_matches

    return SearchResponse(total=len(mp3_files), files=mp3_files)


@app.get("/api/random", response_model=SongInfo)
async def get_random_song(q: str = Query("", description="Filter by search query")):
    """Get a random song, optionally filtered by search query"""
    if not _ID_MAPPING:
        raise HTTPException(status_code=404, detail="No songs found")

    indices_to_choose = list(_ID_MAPPING.keys())

    if q:
        traditional_matches = [idx for idx, rel_path in _ID_MAPPING.items() if is_match(q, rel_path)]

        if len(traditional_matches) == 1:
            indices_to_choose = traditional_matches
        else:
            matched_ids = ai_search(q)
            if not matched_ids:
                indices_to_choose = traditional_matches
            else:
                indices_to_choose = matched_ids

    if not indices_to_choose:
        raise HTTPException(status_code=404, detail="No matching songs found")

    random_idx = random.choice(indices_to_choose)
    rel_path = _ID_MAPPING[random_idx]
    file_path = MUSIC_DIR / rel_path

    return SongInfo(
        id=random_idx,
        name=clean_song_name(file_path.name),
        path=rel_path,
        size=file_path.stat().st_size,
        image=get_image_url(rel_path)
    )


@app.get("/api/songs/{song_id}")
async def get_song(song_id: int):
    """Get song details by ID"""
    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    rel_path = _ID_MAPPING[song_id]
    file_path = MUSIC_DIR / rel_path

    return SongInfo(
        id=song_id,
        name=clean_song_name(file_path.name),
        path=rel_path,
        size=file_path.stat().st_size,
        image=get_image_url(rel_path)
    )


@app.get("/api/stream/{song_id}")
async def stream_song(song_id: int):
    """Stream audio file by ID"""
    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    rel_path = _ID_MAPPING[song_id]
    file_path = MUSIC_DIR / rel_path

    def iterfile():
        with open(file_path, "rb") as f:
            yield from f

    # Encode filename for HTTP header (RFC 5987)
    encoded_filename = quote(file_path.name, safe='')

    return StreamingResponse(
        iterfile(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}",
            "Accept-Ranges": "bytes"
        }
    )


@app.get("/api/download/{song_id}")
async def download_song(song_id: int):
    """Download audio file by ID"""
    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    rel_path = _ID_MAPPING[song_id]
    file_path = MUSIC_DIR / rel_path

    from urllib.parse import quote
    encoded_filename = quote(file_path.name, safe='')

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=file_path.name,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )


@app.get("/api/image/{image_id}")
async def get_image(image_id: int):
    """Get image by ID"""
    if image_id not in _IMAGE_ID_MAPPING:
        raise HTTPException(status_code=404, detail="Image not found")

    relative_path = _IMAGE_ID_MAPPING[image_id]
    local_path = MUSIC_DIR / relative_path

    if not local_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Determine content type
    if relative_path.lower().endswith(".png"):
        media_type = "image/png"
    elif relative_path.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif relative_path.lower().endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        local_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=604800"}
    )


@app.post("/api/image/generate")
async def generate_image(request: ImageGenerateRequest):
    """Generate image using SiliconFlow API"""
    try:
        siliconflow_request = {
            "model": request.model,
            "prompt": request.prompt,
            "image_size": request.image_size,
            "batch_size": request.batch_size,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }

        if request.negative_prompt:
            siliconflow_request["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            siliconflow_request["seed"] = request.seed

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SILICONFLOW_API_URL,
                json=siliconflow_request,
                headers={
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                    "Content-Type": "application/json",
                }
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"SiliconFlow API error: {response.status_code}"
            )

        response_data = response.json()
        images = response_data.get("images", [])
        timings = response_data.get("timings", {})
        generated_seed = response_data.get("seed")

        # Download and cache images
        cached_images = []
        import time
        import hashlib

        for idx, img in enumerate(images):
            external_url = img.get("url")
            filename = f"img_{generated_seed}_{idx}_{int(time.time())}.png"
            local_path = IMAGE_CACHE_DIR / filename

            try:
                img_response = await httpx.AsyncClient().get(external_url, timeout=30)
                img_response.raise_for_status()

                # Load and resize image to 240x240
                image = Image.open(io.BytesIO(img_response.content))
                image_resized = image.resize((240, 240), Image.Resampling.LANCZOS)
                image_resized.save(local_path, "PNG")

                cached_images.append({
                    "url": f"/api/image/get/{filename}",
                    "filename": filename,
                    "index": idx,
                    "size": local_path.stat().st_size
                })
            except Exception as e:
                cached_images.append({
                    "url": external_url,
                    "index": idx,
                    "error": str(e)
                })

        return {
            "success": True,
            "images": cached_images,
            "parameters": {
                "model": request.model,
                "prompt": request.prompt,
                "image_size": request.image_size,
                "batch_size": request.batch_size,
                "seed": generated_seed
            },
            "timings": timings
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout (60s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/image/get/{filename}")
async def get_cached_image(filename: str):
    """Get cached generated image"""
    local_path = IMAGE_CACHE_DIR / filename

    if not local_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        local_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=604800"}
    )


# Playlist endpoints
@app.get("/api/playlists")
async def list_playlists():
    """List all playlists"""
    playlists = []
    for file_path in PLAYLISTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                playlists.append({
                    "id": file_path.stem,
                    "name": data.get("name", file_path.stem),
                    "description": data.get("description", ""),
                    "song_count": len(data.get("songs", [])),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                })
        except Exception:
            continue

    return {"total": len(playlists), "playlists": playlists}


@app.post("/api/playlists")
async def create_playlist(request: PlaylistCreate):
    """Create a new playlist"""
    playlist_id = f"playlist_{int(datetime.now().timestamp())}"

    playlist_data = {
        "id": playlist_id,
        "name": request.name,
        "description": request.description,
        "songs": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(playlist_data, f, ensure_ascii=False, indent=2)

    return playlist_data


@app.get("/api/playlists/{playlist_id}")
async def get_playlist(playlist_id: str):
    """Get playlist details"""
    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add full song info
    songs = []
    for item in data.get("songs", []):
        song_id = item.get("song_id")
        if song_id in _ID_MAPPING:
            rel_path = _ID_MAPPING[song_id]
            file_path_full = MUSIC_DIR / rel_path
            songs.append({
                "id": song_id,
                "name": clean_song_name(file_path_full.name),
                "path": rel_path,
                "size": file_path_full.stat().st_size,
                "image": get_image_url(rel_path),
                "added_at": item.get("added_at")
            })

    data["songs"] = songs
    return data


@app.put("/api/playlists/{playlist_id}")
async def update_playlist(playlist_id: str, request: PlaylistUpdate):
    """Update playlist"""
    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if request.name is not None:
        data["name"] = request.name
    if request.description is not None:
        data["description"] = request.description
    if request.song_ids is not None:
        data["songs"] = [
            {"song_id": sid, "added_at": datetime.now().isoformat()}
            for sid in request.song_ids
        ]

    data["updated_at"] = datetime.now().isoformat()

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


@app.delete("/api/playlists/{playlist_id}")
async def delete_playlist(playlist_id: str):
    """Delete playlist"""
    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    file_path.unlink()
    return {"message": "Playlist deleted"}


@app.post("/api/playlists/{playlist_id}/songs")
async def add_song_to_playlist(playlist_id: str, song_id: int):
    """Add song to playlist"""
    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if song already exists
    for item in data.get("songs", []):
        if item.get("song_id") == song_id:
            return data  # Already in playlist

    data["songs"].append({
        "song_id": song_id,
        "added_at": datetime.now().isoformat()
    })
    data["updated_at"] = datetime.now().isoformat()

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


@app.delete("/api/playlists/{playlist_id}/songs/{song_id}")
async def remove_song_from_playlist(playlist_id: str, song_id: int):
    """Remove song from playlist"""
    file_path = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["songs"] = [s for s in data.get("songs", []) if s.get("song_id") != song_id]
    data["updated_at"] = datetime.now().isoformat()

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


# Favorites endpoints
FAVORITES_FILE = PLAYLISTS_DIR / "favorites.json"

@app.get("/api/favorites")
async def get_favorites():
    """Get favorite songs"""
    if not FAVORITES_FILE.exists():
        return {"songs": []}

    with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    songs = []
    for song_id in data.get("songs", []):
        if song_id in _ID_MAPPING:
            rel_path = _ID_MAPPING[song_id]
            file_path_full = MUSIC_DIR / rel_path
            songs.append({
                "id": song_id,
                "name": clean_song_name(file_path_full.name),
                "path": rel_path,
                "size": file_path_full.stat().st_size,
                "image": get_image_url(rel_path)
            })

    return {"songs": songs}


@app.post("/api/favorites/{song_id}")
async def add_favorite(song_id: int):
    """Add song to favorites"""
    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    favorites = {"songs": []}
    if FAVORITES_FILE.exists():
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            favorites = json.load(f)

    if song_id not in favorites["songs"]:
        favorites["songs"].append(song_id)

    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)

    return favorites


@app.delete("/api/favorites/{song_id}")
async def remove_favorite(song_id: int):
    """Remove song from favorites"""
    if not FAVORITES_FILE.exists():
        return {"songs": []}

    with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
        favorites = json.load(f)

    favorites["songs"] = [sid for sid in favorites["songs"] if sid != song_id]

    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)

    return favorites


# History endpoints
HISTORY_FILE = PLAYLISTS_DIR / "history.json"

@app.get("/api/history")
async def get_history(limit: int = Query(50, description="Max number of records")):
    """Get play history"""
    if not HISTORY_FILE.exists():
        return {"history": []}

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])[:limit]

    songs = []
    for item in history:
        song_id = item.get("song_id")
        if song_id in _ID_MAPPING:
            rel_path = _ID_MAPPING[song_id]
            file_path_full = MUSIC_DIR / rel_path
            songs.append({
                "id": song_id,
                "name": clean_song_name(file_path_full.name),
                "path": rel_path,
                "size": file_path_full.stat().st_size,
                "image": get_image_url(rel_path),
                "played_at": item.get("played_at")
            })

    return {"history": songs}


@app.post("/api/history/{song_id}")
async def add_to_history(song_id: int):
    """Add song to play history"""
    if song_id not in _ID_MAPPING:
        raise HTTPException(status_code=404, detail="Song not found")

    history = {"history": []}
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

    # Add to beginning
    history["history"].insert(0, {
        "song_id": song_id,
        "played_at": datetime.now().isoformat()
    })

    # Keep only last 100
    history["history"] = history["history"][:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return history


@app.delete("/api/history")
async def clear_history():
    """Clear play history"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"history": []}, f)

    return {"message": "History cleared"}


# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle messages
            for connection in active_connections:
                if connection != websocket:
                    await connection.send_text(data)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# Mount static files for serving the web client
@app.get("/")
async def root():
    """Serve the web client"""
    return FileResponse("static/index.html")


# Mount static directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def main():
    """Main entry point"""
    # Build ID mapping
    build_id_mapping()

    server_ip = SERVER_IP if SERVER_IP else get_local_ip()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║              FastAPI Music Server v2.0                      ║
╠════════════════════════════════════════════════════════════╣
║  Server: http://{server_ip}:{PORT}                    ║
║  Music Directory: {str(MUSIC_DIR):<42} ║
║  Total Songs: {len(_ID_MAPPING):<45} ║
║  Total Images: {len(_IMAGE_ID_MAPPING):<43} ║
╠════════════════════════════════════════════════════════════╣
║  API Endpoints:                                            ║
║    GET  /api/health           - Health check               ║
║    GET  /api/list             - List all songs             ║
║    GET  /api/search?q=        - Search songs               ║
║    GET  /api/random           - Random song                ║
║    GET  /api/songs/{{id}}       - Song details               ║
║    GET  /api/stream/{{id}}      - Stream audio               ║
║    GET  /api/download/{{id}}    - Download audio             ║
║    GET  /api/image/{{id}}       - Get image                  ║
║    POST /api/image/generate    - Generate image             ║
╠════════════════════════════════════════════════════════════╣
║  Playlist Endpoints:                                        ║
║    GET    /api/playlists         - List playlists           ║
║    POST   /api/playlists         - Create playlist          ║
║    GET    /api/playlists/{{id}}    - Get playlist             ║
║    PUT    /api/playlists/{{id}}    - Update playlist          ║
║    DELETE /api/playlists/{{id}}    - Delete playlist          ║
╠════════════════════════════════════════════════════════════╣
║  Library Endpoints:                                         ║
║    GET    /api/favorites         - Get favorites            ║
║    POST   /api/favorites/{{id}}    - Add to favorites         ║
║    DELETE /api/favorites/{{id}}    - Remove from favorites    ║
║    GET    /api/history           - Get play history          ║
║    POST   /api/history/{{id}}      - Add to history           ║
║    DELETE /api/history           - Clear history            ║
╠════════════════════════════════════════════════════════════╣
║  WebSocket:                                                ║
║    WS /ws                     - Real-time updates          ║
╚════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
