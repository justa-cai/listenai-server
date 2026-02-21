#!/usr/bin/env python3
"""
HTTP Server for MP3 Resources & Image Generation
Provides static file serving, music search API, and image generation API
"""

import http.server
import socketserver
import json
import random
import re
import os
import ast
import requests
import socket
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote
from openai import OpenAI
from PIL import Image
import io
from urllib.parse import quote

PORT = 9100
SERVER_IP = ""
MUSIC_DIR = Path(__file__).parent / "data"

# OpenAI AI Search Configuration
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://192.168.13.228:8000/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "123")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen3-30B-A3B")

# Initialize OpenAI client
ai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# SiliconFlow Image Generation Configuration
SILICONFLOW_API_KEY = "sk-qnshhbcisgilevxquheykzdblutjagkjyfrodbntldipcbvg"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/images/generations"

# Image cache directory
IMAGE_CACHE_DIR = MUSIC_DIR / "cache" / "images"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global ID mapping - initialized once at startup
_ID_MAPPING = None
_REVERSE_MAPPING = None
_IMAGE_ID_MAPPING = None
_IMAGE_REVERSE_MAPPING = None


def get_local_ip():
    """Get local IP address automatically"""
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
    # Remove numeric prefix (e.g., "123." or "001.")
    name = re.sub(r"^\d+\.\s*", "", filename)
    # Remove .mp3 suffix
    name = name.replace(".mp3", "")
    return name


def normalize_text(text):
    """Normalize text for better search matching"""
    # Replace path separators and brackets with space
    text = re.sub(r"[\/《》\[\]()\-·\.]+", " ", text)
    # Remove special characters
    text = re.sub(r"[^\w\s\u4e00-\u9fff]+", "", text)
    # Remove common meaningless words
    text = text.replace("周杰伦", "")
    text = text.replace("的", "")
    text = text.replace("、", "")
    return text.strip().lower()


def ai_search(search_query):
    """Use AI model to find matching songs"""
    if not _ID_MAPPING:
        return []

    try:
        # Prepare song list (limit to 500 for efficiency)
        songs_list = []
        max_songs = min(500, len(_ID_MAPPING))

        for idx, rel_path in list(_ID_MAPPING.items())[:max_songs]:
            file_path = MUSIC_DIR / rel_path
            songs_list.append(
                {"id": idx, "name": clean_song_name(file_path.name), "path": rel_path}
            )

        # Create prompt for AI
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

        # Call AI model
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

        print(f"[DEBUG] AI Request JSON:")
        request_summary = {
            "model": request_data["model"],
            "temperature": request_data["temperature"],
            "max_tokens": request_data["max_tokens"],
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"][:200] + "..."
                    if len(msg["content"]) > 200
                    else msg["content"],
                }
                for msg in request_data["messages"]
            ],
            "songs_count": len(songs_list),
        }
        print(json.dumps(request_summary, ensure_ascii=False, indent=2))

        response = ai_client.chat.completions.create(**request_data)

        # Parse AI response
        ai_response = response.choices[0].message.content.strip()

        print(f"[DEBUG] AI Response JSON:")
        response_data = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        print(json.dumps(response_data, ensure_ascii=False, indent=2))

        # Try to extract JSON array from response
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

        print(f"[DEBUG] AI response not in expected format: {ai_response[:200]}")
        return []

    except Exception as e:
        print(f"[DEBUG] AI search failed: {type(e).__name__}: {e}")
        return []


def is_match(search_query, file_path):
    """Check if search query matches file path using intelligent matching"""
    search_lower = search_query.lower()

    # Direct match
    if search_lower in file_path.lower():
        return True

    # Normalize and check
    normalized_query = normalize_text(search_lower)
    normalized_path = normalize_text(file_path)

    if not normalized_query:
        return True

    if normalized_query in normalized_path:
        return True

    # Split query into keywords and check if all keywords match
    keywords = re.split(r"[\s\-·、]+", normalized_query)
    keywords = [k for k in keywords if k]

    if not keywords:
        return True

    for keyword in keywords:
        if keyword not in normalized_path:
            return False

    return True


def build_id_mapping():
    """Build ID to filename mapping from MP3 files and PNG images"""
    global _ID_MAPPING, _REVERSE_MAPPING, _IMAGE_ID_MAPPING, _IMAGE_REVERSE_MAPPING
    mapping = {}
    idx = 1
    for file_path in sorted(MUSIC_DIR.glob("**/*.mp3")):
        # Skip backup, temporary directories and hidden files
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

    # Build image ID mapping for PNG files (same directory as MP3)
    image_mapping = {}
    img_idx = 1
    for file_path in sorted(MUSIC_DIR.glob("**/*.png")):
        # Skip backup, temporary directories, hidden files and Z-Image-Turbo assets
        if (
            "backup_original" in file_path.parts
            or ".tmp" in str(file_path)
            or "_temp" in str(file_path)
            or "/.venv/" in str(file_path)
            or "/__pycache__/" in str(file_path)
            or "/.pytest_cache/" in str(file_path)
            or ".git" in str(file_path).lower()
            or "assets" in str(file_path).lower()
        ):
            continue
        image_mapping[img_idx] = str(file_path.relative_to(MUSIC_DIR))
        img_idx += 1
    _IMAGE_ID_MAPPING = image_mapping
    _IMAGE_REVERSE_MAPPING = {v: k for k, v in image_mapping.items()}


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
                return f"http://{server_ip}:{PORT}/api/image/{image_id}"
        return None
    except Exception:
        return None
    except Exception:
        return None


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MP3RequestHandler(http.server.SimpleHTTPRequestHandler):
    # Force HTTP/1.0 to avoid chunked encoding for embedded players
    protocol_version = "HTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(MUSIC_DIR), **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Normalize path to handle double slashes and leading/trailing slashes
        path = "/" + "/".join([p for p in path.split("/") if p])

        # Music API endpoints
        if path == "/api/list" or path.startswith("/api/list?"):
            self.handle_list_api(parsed_path)
        elif path == "/api/search" or path.startswith("/api/search?"):
            self.handle_search_api(parsed_path)
        elif path == "/api/random" or path.startswith("/api/random?"):
            self.handle_random_api(parsed_path)
        elif path.startswith("/api/download/"):
            self.handle_download_api(path)
        elif path.startswith("/api/image/"):
            self.handle_image_get_api(path)
        else:
            # Serve static files (MP3s)
            super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Normalize path
        path = "/" + "/".join([p for p in path.split("/") if p])

        # Image generation endpoint
        if path == "/api/image/generate":
            self.handle_image_generate_api(parsed_path)
        else:
            # Serve static files
            super().do_GET()

    def handle_list_api(self, parsed_path):
        """Handle list API endpoint"""
        query_params = parse_qs(parsed_path.query)
        search_query = query_params.get("q", [""])[0].lower()

        # Collect MP3 files info with short ID
        mp3_files = []
        for idx, rel_path in _ID_MAPPING.items():
            file_path = MUSIC_DIR / rel_path

            # Filter by search query if provided
            if search_query and search_query not in rel_path.lower():
                continue

            info = {
                "id": idx,
                "name": clean_song_name(file_path.name),
                "size": file_path.stat().st_size,
                "image": get_image_url(rel_path),
            }

            mp3_files.append(info)

        response = {"total": len(mp3_files), "files": mp3_files}

        self.send_json_response(response)

    def handle_search_api(self, parsed_path):
        """Handle search API endpoint using hybrid search"""
        query_params = parse_qs(parsed_path.query)
        search_query = query_params.get("q", [""])[0]

        print(f"[DEBUG] Search query: '{search_query}'")
        print(
            f"[DEBUG] Total files in mapping: {len(_ID_MAPPING) if _ID_MAPPING else 0}"
        )

        if not search_query:
            self.send_json_response(
                {"error": "Search query parameter 'q' is required"}, 400
            )
            return

        mp3_files = []

        # Step 1: Try traditional search first
        print(f"[DEBUG] Step 1: Traditional search")
        traditional_matches = []
        for idx, rel_path in _ID_MAPPING.items():
            file_path = MUSIC_DIR / rel_path
            if is_match(search_query, rel_path):
                info = {
                    "id": idx,
                    "name": clean_song_name(file_path.name),
                    "size": file_path.stat().st_size,
                    "image": get_image_url(rel_path),
                }
                traditional_matches.append(info)

        print(f"[DEBUG] Traditional search found {len(traditional_matches)} matches")

        # Step 2: Decision based on traditional search results
        if len(traditional_matches) == 1:
            # Perfect match found, return directly
            print(f"[DEBUG] Single match found, returning without AI")
            mp3_files = traditional_matches
        else:
            # No match or multiple matches, use AI for better results
            if len(traditional_matches) == 0:
                print(f"[DEBUG] No traditional matches, using AI search")
            else:
                print(
                    f"[DEBUG] Multiple traditional matches ({len(traditional_matches)}), using AI search"
                )

            # Use AI search
            matched_ids = ai_search(search_query)

            if matched_ids:
                print(f"[DEBUG] AI found {len(matched_ids)} matches")
                for idx in matched_ids:
                    rel_path = _ID_MAPPING[idx]
                    file_path = MUSIC_DIR / rel_path
                    info = {
                        "id": idx,
                        "name": clean_song_name(file_path.name),
                        "size": file_path.stat().st_size,
                        "image": get_image_url(rel_path),
                    }
                    mp3_files.append(info)
            else:
                print(
                    f"[DEBUG] AI search returned no results, fallback to traditional results"
                )
                mp3_files = traditional_matches

        print(f"[DEBUG] Found {len(mp3_files)} matches")

        response = {"total": len(mp3_files), "files": mp3_files}

        self.send_json_response(response)

    def handle_random_api(self, parsed_path):
        """Handle random song API endpoint using hybrid search"""
        if not _ID_MAPPING:
            self.send_json_response({"error": "No MP3 files found"}, 404)
            return

        query_params = parse_qs(parsed_path.query)
        search_query = query_params.get("q", [""])[0]

        # Filter by search query if provided using hybrid search
        if search_query:
            print(f"[DEBUG] Random API search query: '{search_query}'")

            # Step 1: Try traditional search first
            traditional_matches = []
            for idx, rel_path in _ID_MAPPING.items():
                if is_match(search_query, rel_path):
                    traditional_matches.append(idx)

            print(
                f"[DEBUG] Random API traditional search found {len(traditional_matches)} matches"
            )

            # Step 2: Decision based on traditional search results
            if len(traditional_matches) == 1:
                # Perfect match found, use directly
                print(f"[DEBUG] Random API single match found, returning without AI")
                indices_to_choose = traditional_matches
            else:
                # No match or multiple matches, use AI for better results
                if len(traditional_matches) == 0:
                    print(f"[DEBUG] Random API no traditional matches, using AI search")
                else:
                    print(
                        f"[DEBUG] Random API multiple traditional matches ({len(traditional_matches)}), using AI search"
                    )

                matched_ids = ai_search(search_query)
                if not matched_ids:
                    print(
                        f"[DEBUG] Random API AI search returned no results, fallback to traditional"
                    )
                    indices_to_choose = traditional_matches
                else:
                    indices_to_choose = matched_ids
        else:
            indices_to_choose = list(_ID_MAPPING.keys())

        if not indices_to_choose:
            self.send_json_response({"id": None, "name": None, "size": None})
            return

        random_idx = random.choice(indices_to_choose)
        rel_path = _ID_MAPPING[random_idx]
        file_path = MUSIC_DIR / rel_path

        response = {
            "id": random_idx,
            "name": clean_song_name(file_path.name),
            "size": file_path.stat().st_size,
            "image": get_image_url(rel_path),
        }

        self.send_json_response(response)

    def handle_download_api(self, path):
        """Handle download API endpoint with short ID"""
        try:
            print(f"[DEBUG] Original path: '{path}'")

            # Clean path - extract the ID part only
            # Handle cases like: /api/download/58, /api/download/58 HTTP/1.1
            path_parts = path.strip().split()
            if path_parts:
                clean_path = path_parts[0]
            else:
                clean_path = path

            print(f"[DEBUG] Clean path: '{clean_path}'")

            # Extract ID from path like /api/download/123
            song_id = int(clean_path.split("/")[-1])

            print(f"[DEBUG] Extracted song_id: {song_id}")

            rel_path = _ID_MAPPING.get(song_id)
            if rel_path:
                file_path = MUSIC_DIR / rel_path
                print(f"[DEBUG] Song name: {clean_song_name(file_path.name)}")

            if song_id not in _ID_MAPPING:
                self.send_json_response({"error": "Song not found"}, 404)
                return

            rel_path = _ID_MAPPING[song_id]
            file_path = MUSIC_DIR / rel_path

            # Serve the actual MP3 file
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            encoded_filename = quote(file_path.name)
            self.send_header(
                "Content-Disposition",
                f"attachment; filename*=UTF-8''{encoded_filename}",
            )
            self.end_headers()
            self.wfile.write(content)
            return  # Important: exit after sending file
        except (BrokenPipeError, ConnectionResetError):
            print(f"[INFO] Client disconnected during download (ID: {song_id})")
        except (ValueError, IndexError) as e:
            print(f"[DEBUG] Exception occurred: {type(e).__name__}: {e}")
            print(f"[DEBUG] Path was: '{path}'")
            self.send_json_response({"error": "Invalid song ID"}, 400)

    def handle_image_generate_api(self, parsed_path):
        """Handle image generation API"""
        try:
            # Parse request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_json_response({"error": "Request body is required"}, 400)
                return

            request_data = self.rfile.read(content_length)
            body = json.loads(request_data.decode("utf-8"))

            # Extract parameters
            prompt = body.get("prompt")
            if not prompt:
                self.send_json_response(
                    {"error": "'prompt' parameter is required"}, 400
                )
                return

            model = body.get("model", "Qwen/Qwen-Image-Edit-2509")
            negative_prompt = body.get("negative_prompt", "")
            image_size = body.get("image_size", "1024x1024")
            batch_size = body.get("batch_size", 1)
            seed = body.get("seed")
            num_inference_steps = body.get("num_inference_steps", 20)
            guidance_scale = body.get("guidance_scale", 7.5)

            print(f"[DEBUG] Image generation request:")
            print(f"  Model: {model}")
            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Size: {image_size}")
            print(f"  Batch: {batch_size}")

            # Call SiliconFlow API
            siliconflow_request = {
                "model": model,
                "prompt": prompt,
                "image_size": image_size,
                "batch_size": batch_size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }

            if negative_prompt:
                siliconflow_request["negative_prompt"] = negative_prompt
            if seed is not None:
                siliconflow_request["seed"] = seed

            print(f"[DEBUG] Calling SiliconFlow API...")
            response = requests.post(
                SILICONFLOW_API_URL,
                json=siliconflow_request,
                headers={
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )

            if response.status_code != 200:
                print(f"[ERROR] SiliconFlow API error: {response.status_code}")
                print(f"[ERROR] Response: {response.text[:500]}")
                self.send_json_response(
                    {
                        "error": f"SiliconFlow API error: {response.status_code}",
                        "detail": response.text[:200],
                    },
                    500,
                )
                return

            response_data = response.json()
            images = response_data.get("images", [])
            timings = response_data.get("timings", {})
            generated_seed = response_data.get("seed")

            print(f"[DEBUG] Generated {len(images)} images")
            print(f"[DEBUG] Inference time: {timings.get('inference', 'N/A')}s")

            # Download and cache images
            cached_images = []
            import hashlib
            import time

            for idx, img in enumerate(images):
                external_url = img.get("url")

                # Generate unique filename based on seed and index
                filename = f"img_{generated_seed}_{idx}_{int(time.time())}.png"
                local_path = IMAGE_CACHE_DIR / filename

                print(f"[DEBUG] Downloading image {idx}: {external_url}")
                print(f"[DEBUG] Caching to: {local_path}")

                try:
                    img_response = requests.get(external_url, timeout=30)
                    img_response.raise_for_status()

                    # Load and resize image to 240x240
                    img = Image.open(io.BytesIO(img_response.content))
                    img_resized = img.resize((240, 240), Image.Resampling.LANCZOS)

                    # Save resized image
                    img_resized.save(local_path, "PNG")

                    print(f"[DEBUG] Image resized to 240x240 and cached: {filename}")

                    server_ip = SERVER_IP if SERVER_IP else get_local_ip()
                    cached_images.append(
                        {
                            "url": f"http://{server_ip}:{PORT}/api/image/get/{filename}",
                            "filename": filename,
                            "index": idx,
                            "size": local_path.stat().st_size,
                        }
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to cache image {idx}: {e}")
                    cached_images.append(
                        {"url": external_url, "index": idx, "error": str(e)}
                    )

            # Format response
            result = {
                "success": True,
                "images": cached_images,
                "parameters": {
                    "model": model,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": generated_seed,
                },
                "timings": timings,
            }

            self.send_json_response(result)

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {e}")
            self.send_json_response({"error": "Invalid JSON in request body"}, 400)
        except requests.Timeout:
            print(f"[ERROR] Request timeout")
            self.send_json_response({"error": "Request timeout (60s)"}, 504)
        except requests.RequestException as e:
            print(f"[ERROR] Request exception: {e}")
            self.send_json_response({"error": f"Request failed: {str(e)}"}, 500)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            self.send_json_response({"error": f"Internal server error: {str(e)}"}, 500)

    def handle_image_get_api(self, path):
        """Handle image retrieval API by short ID"""
        try:
            # Extract image ID from path like /api/image/123
            # Remove leading slash and split by /
            path = path.strip("/")
            if not path.startswith("api/image/"):
                self.send_json_response({"error": "Invalid image path"}, 400)
                return

            # Extract ID from path like /api/image/123
            parts = path.split("/")
            if len(parts) < 3:
                self.send_json_response({"error": "Invalid image path"}, 400)
                return

            image_id_str = parts[2]
            try:
                image_id = int(image_id_str)
            except ValueError:
                self.send_json_response({"error": "Invalid image ID"}, 400)
                return

            # Get image path from ID mapping
            relative_path = _IMAGE_ID_MAPPING.get(image_id)
            if not relative_path:
                self.send_json_response({"error": "Image not found"}, 404)
                return

            local_path = MUSIC_DIR / relative_path

            print(f"[DEBUG] Image get request: ID={image_id}")
            print(f"[DEBUG] Local path: {local_path}")

            if not local_path.exists():
                self.send_json_response({"error": "Image not found"}, 404)
                return

            # Determine content type based on file extension
            if relative_path.lower().endswith(".png"):
                content_type = "image/png"
            elif relative_path.lower().endswith((".jpg", ".jpeg")):
                content_type = "image/jpeg"
            elif relative_path.lower().endswith(".webp"):
                content_type = "image/webp"
            else:
                content_type = "application/octet-stream"

            # Serve image
            with open(local_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", f"public, max-age={7 * 24 * 3600}")
            self.end_headers()
            self.wfile.write(content)
            return

        except Exception as e:
            print(f"[ERROR] Image get error: {type(e).__name__}: {e}")
            self.send_json_response({"error": f"Failed to serve image: {str(e)}"}, 500)

    def send_json_response(self, data, status_code=200):
        """Send a JSON response"""
        content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    # Build ID mapping before starting server
    build_id_mapping()

    server_ip = SERVER_IP if SERVER_IP else get_local_ip()

    print(f"""
╔════════════════════════════════════════════════════╗
║         MP3 Music & Image Server                      ║
╠════════════════════════════════════════════════════╣
║  IP: {server_ip:<50} ║
║  Port: {PORT:<48} ║
║  Directory: {str(MUSIC_DIR):<43} ║
║  Image Cache: {str(IMAGE_CACHE_DIR):<40} ║
╠════════════════════════════════════════════════════╣
║  Music API Endpoints:                                   ║
║    GET /api/list          - List all MP3 files          ║
║    GET /api/search?q=查询 - Search (混合智能搜索)         ║
║    GET /api/random         - Get random song            ║
║    GET /api/download/ID    - Download by short ID        ║
╠════════════════════════════════════════════════════╣
║  Image API Endpoints:                                  ║
║    GET  /api/image/get/<path>  - Get image file          ║
╠════════════════════════════════════════════════════╣
║  Search Strategy:                                         ║
║    1. Traditional search (fast)                            ║
║    2. If 1 match → Return directly                         ║
║    3. If 0/multiple matches → Use AI search              ║
╠════════════════════════════════════════════════════╣
║  AI Search Config:                                       ║
║    API Base: {OPENAI_API_BASE:<42} ║
║    Model: {OPENAI_MODEL:<45} ║
╠════════════════════════════════════════════════════╣
║  Image Config:                                          ║
║    Source: MP3 directory (auto-generated)                   ║
║    Cache: 7 days                                          ║
╚════════════════════════════════════════════════════╝
    """)

    with ThreadedTCPServer(("0.0.0.0", PORT), MP3RequestHandler) as httpd:
        print(f"Server started at http://{server_ip}:{PORT}")
        print("Press Ctrl+C to stop the server\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
