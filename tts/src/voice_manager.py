"""Voice manager for handling voice cloning samples."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Voice information."""
    id: str  # voice_id (e.g., "视频配音-和蔼奶奶")
    name: str  # voice name (e.g., "和蔼奶奶")
    category: str  # category (e.g., "视频配音")
    audio_path: str  # full path to mp3 file
    text_path: str  # full path to txt file
    sample_text: str  # sample text content
    created_at: float  # timestamp


class VoiceManager:
    """Manager for voice cloning samples."""

    def __init__(self, voice_dir: str = None):
        """
        Initialize the voice manager.

        Args:
            voice_dir: Base directory containing voice samples
        """
        self.voice_dir = Path(voice_dir) if voice_dir else None
        self.voices: Dict[str, VoiceInfo] = {}
        self.categories: Dict[str, List[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize voice manager by scanning voice directory."""
        if not self.voice_dir:
            logger.warning("Voice directory not configured")
            return

        if not self.voice_dir.exists():
            logger.warning(f"Voice directory not found: {self.voice_dir}")
            return

        logger.info(f"Scanning voice directory: {self.voice_dir}")
        await self._scan_voices()
        self._initialized = True
        logger.info(f"Voice manager initialized with {len(self.voices)} voices from {len(self.categories)} categories")

        # Log categories and counts
        for category, voices in sorted(self.categories.items()):
            logger.info(f"  - {category}: {len(voices)} voices")

    async def _scan_voices(self) -> None:
        """Scan voice directory and load voice information."""
        self.voices.clear()
        self.categories.clear()

        if not self.voice_dir:
            return

        # Use recursive search to find all mp3 files
        for mp3_path in self.voice_dir.rglob("*.mp3"):
            # Get relative path from voice_dir
            rel_path = mp3_path.relative_to(self.voice_dir)
            path_parts = rel_path.parts

            # The category is the parent directory name
            # For voice_clone/doubao/视频配音/voice.mp3:
            # - rel_path = doubao/视频配音/voice.mp3
            # - We want "视频配音" as category
            if len(path_parts) >= 2:
                category = path_parts[-2]  # Parent directory name
            else:
                category = "默认"  # Fallback

            voice_name = mp3_path.stem  # filename without extension
            txt_path = mp3_path.with_suffix(".txt")

            # Read sample text
            sample_text = ""
            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        sample_text = f.read().strip()
                except Exception as e:
                    logger.warning(f"Failed to read {txt_path}: {e}")

            # Create voice ID
            voice_id = f"{category}-{voice_name}"

            # Create voice info
            voice_info = VoiceInfo(
                id=voice_id,
                name=voice_name,
                category=category,
                audio_path=str(mp3_path),
                text_path=str(txt_path),
                sample_text=sample_text,
                created_at=mp3_path.stat().st_mtime
            )

            self.voices[voice_id] = voice_info

            # Track categories
            if category not in self.categories:
                self.categories[category] = []
            if voice_name not in self.categories[category]:
                self.categories[category].append(voice_name)

    def get_voice(self, voice_id: str) -> Optional[VoiceInfo]:
        """
        Get voice information by ID.

        Args:
            voice_id: Voice ID in format "category-voice_name"

        Returns:
            VoiceInfo if found, None otherwise
        """
        return self.voices.get(voice_id)

    def get_voice_path(self, voice_id: str) -> Optional[str]:
        """
        Get audio file path for a voice.

        Args:
            voice_id: Voice ID in format "category-voice_name"

        Returns:
            Path to audio file if found, None otherwise
        """
        voice = self.get_voice(voice_id)
        return voice.audio_path if voice else None

    def list_voices(self, category: str = None) -> List[VoiceInfo]:
        """
        List available voices.

        Args:
            category: Optional category filter

        Returns:
            List of VoiceInfo objects
        """
        if category:
            return [v for v in self.voices.values() if v.category == category]
        return list(self.voices.values())

    def list_categories(self) -> List[str]:
        """
        List available categories.

        Returns:
            List of category names
        """
        return list(self.categories.keys())

    def get_voices_dict(self, category: str = None) -> Dict[str, List[Dict]]:
        """
        Get voices organized by category as dictionary.

        Args:
            category: Optional category filter

        Returns:
            Dictionary mapping categories to voice lists
        """
        result = {}

        for voice in self.voices.values():
            if category and voice.category != category:
                continue

            if voice.category not in result:
                result[voice.category] = []

            result[voice.category].append({
                "id": voice.id,
                "name": voice.name,
                "sample_text": voice.sample_text,
                "audio_url": f"/api/voices/{voice.id}/audio"
            })

        # Sort voices by name within each category
        for cat in result:
            result[cat].sort(key=lambda x: x["name"])

        return result

    def search_voices(self, query: str) -> List[VoiceInfo]:
        """
        Search voices by name or sample text.

        Args:
            query: Search query

        Returns:
            List of matching VoiceInfo objects
        """
        query_lower = query.lower()
        results = []

        for voice in self.voices.values():
            if (query_lower in voice.name.lower() or
                query_lower in voice.sample_text.lower() or
                query_lower in voice.category.lower()):
                results.append(voice)

        return results

    def get_stats(self) -> Dict:
        """
        Get voice manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_voices": len(self.voices),
            "total_categories": len(self.categories),
            "voices_per_category": {
                cat: len(voices) for cat, voices in self.categories.items()
            },
            "initialized": self._initialized
        }


# Global voice manager instance
_voice_manager: Optional[VoiceManager] = None


async def get_voice_manager(voice_dir: str = None) -> VoiceManager:
    """
    Get the global voice manager instance.

    Args:
        voice_dir: Optional voice directory path

    Returns:
        VoiceManager instance
    """
    global _voice_manager

    if _voice_manager is None:
        _voice_manager = VoiceManager(voice_dir)
        await _voice_manager.initialize()

    return _voice_manager


def set_voice_manager(manager: VoiceManager) -> None:
    """Set the global voice manager instance."""
    global _voice_manager
    _voice_manager = manager
