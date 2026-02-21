"""HTTP server for serving the web client interface."""

import os
import asyncio
import logging
from aiohttp import web
from typing import Optional

logger = logging.getLogger(__name__)


class HTTPWebServer:
    """
    HTTP server for serving the VoxCPM TTS web client.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9301, voice_manager=None):
        """
        Initialize the HTTP web server.

        Args:
            host: Host to bind to
            port: Port to bind to
            voice_manager: VoiceManager instance for voice APIs
        """
        self.host = host
        self.port = port
        self.voice_manager = voice_manager
        self.app = web.Application()
        self._setup_routes()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get("/", self._serve_index)
        self.app.router.add_get("/api/config", self._serve_config)

        # Voice API routes
        self.app.router.add_get("/api/voices", self._list_voices)
        self.app.router.add_get("/api/voices/categories", self._list_categories)
        self.app.router.add_get("/api/voices/stats", self._voice_stats)
        self.app.router.add_get("/api/voices/{voice_id}/audio", self._serve_voice_audio)

        self.app.router.add_static("/static", self._get_static_path(), name="static")

    def set_voice_manager(self, voice_manager):
        """Set the voice manager."""
        self.voice_manager = voice_manager

    def _get_static_path(self) -> str:
        """Get the path to static files."""
        # Try multiple possible locations for the web directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Check relative to this file (../web/)
        web_dir = os.path.join(os.path.dirname(current_dir), "web")
        if os.path.exists(web_dir):
            return web_dir

        # Check in the parent directory
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        web_dir = os.path.join(parent_dir, "web")
        if os.path.exists(web_dir):
            return web_dir

        # Fallback: use current directory
        return os.path.dirname(current_dir)

    async def _serve_index(self, request: web.Request) -> web.Response:
        """Serve the main HTML page."""
        html_path = os.path.join(self._get_static_path(), "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            return web.Response(text=content, content_type="text/html")
        return web.Response(text="Web client not found", status=404)

    async def _serve_config(self, request: web.Request) -> web.Response:
        """Serve configuration for the web client."""
        import json
        config = {
            "wsUrl": f"ws://{request.host.replace(str(self.port), str(self.port - 1))}/tts",
            "defaultParams": {
                "mode": "streaming",
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "normalize": False,
                "denoise": False
            }
        }
        return web.json_response(config)

    async def _list_voices(self, request: web.Request) -> web.Response:
        """List available voices."""
        if not self.voice_manager:
            return web.json_response({"error": "Voice manager not initialized"}, status=503)

        category = request.query.get("category")
        search = request.query.get("search")

        if search:
            voices = self.voice_manager.search_voices(search)
            voices_dict = [self._voice_to_dict(v) for v in voices]
        else:
            voices_dict = self.voice_manager.get_voices_dict(category)

        return web.json_response({
            "voices": voices_dict,
            "total": len(self.voice_manager.voices)
        })

    async def _list_categories(self, request: web.Request) -> web.Response:
        """List voice categories."""
        if not self.voice_manager:
            return web.json_response({"error": "Voice manager not initialized"}, status=503)

        categories = []
        for cat, voices in self.voice_manager.categories.items():
            categories.append({
                "name": cat,
                "count": len(voices)
            })

        categories.sort(key=lambda x: x["name"])
        return web.json_response({"categories": categories})

    async def _voice_stats(self, request: web.Request) -> web.Response:
        """Get voice statistics."""
        if not self.voice_manager:
            return web.json_response({"error": "Voice manager not initialized"}, status=503)

        return web.json_response(self.voice_manager.get_stats())

    async def _serve_voice_audio(self, request: web.Request) -> web.Response:
        """Serve voice audio file."""
        if not self.voice_manager:
            return web.json_response({"error": "Voice manager not initialized"}, status=503)

        voice_id = request.match_info["voice_id"]
        voice = self.voice_manager.get_voice(voice_id)

        if not voice:
            return web.json_response({"error": "Voice not found"}, status=404)

        audio_path = voice.audio_path
        if not os.path.exists(audio_path):
            return web.json_response({"error": "Audio file not found"}, status=404)

        # Serve the audio file
        with open(audio_path, "rb") as f:
            content = f.read()

        return web.Response(
            body=content,
            content_type="audio/mpeg"
        )

    def _voice_to_dict(self, voice) -> dict:
        """Convert VoiceInfo to dictionary."""
        return {
            "id": voice.id,
            "name": voice.name,
            "category": voice.category,
            "sample_text": voice.sample_text,
            "audio_url": f"/api/voices/{voice.id}/audio"
        }

    async def start(self):
        """Start the HTTP server."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info(f"HTTP web server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP server."""
        if self._runner:
            try:
                await asyncio.wait_for(self._runner.cleanup(), timeout=1.0)
                logger.info("HTTP web server stopped")
            except asyncio.TimeoutError:
                logger.warning("HTTP server cleanup timed out")

    def get_url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"
