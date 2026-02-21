"""Main entry point for VoxCPM WebSocket TTS Server."""

import asyncio
import signal
import logging
import os
import argparse
from pathlib import Path

from .config import Config
from .server import VoxCPMWebSocketServer
from .http_server import HTTPWebServer
from .voice_manager import get_voice_manager
from .logging_config import setup_logging
from .metrics import init_metrics
from .model_cache import ModelCache

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VoxCPM WebSocket TTS Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Enable debug audio saving to ./tmp/debug_audio/"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = Config.from_env()

    # Override config with command line arguments
    if args.debug_audio:
        config.server.debug_audio = True

    # Setup logging
    setup_logging(config.server.log_level, config.server.log_format)
    logger.info("Starting VoxCPM TTS Server")

    # Log configuration
    logger.info(f"WebSocket server: ws://{config.server.host}:{config.server.port}/tts")
    logger.info(f"Model config: name={config.model.model_name}, device={config.model.device}")
    logger.info(f"Max concurrent requests: {config.server.max_concurrent_requests}")
    logger.info(f"Max connections: {config.server.max_connections}")
    logger.info(f"Debug audio: {config.server.debug_audio}")

    # Initialize metrics
    init_metrics(config.server)

    # Load model at startup
    logger.info("Loading VoxCPM model...")
    model_cache = await ModelCache.get_instance()
    model = await model_cache.get_model(config.model)
    logger.info(f"Model loaded successfully: {config.model.model_name}")

    # Resolve voice directory path
    voice_dir = config.server.voice_dir
    if not os.path.isabs(voice_dir):
        # Make relative to the current working directory
        voice_dir = os.path.abspath(voice_dir)

    logger.info(f"Voice directory: {voice_dir}")

    # Initialize voice manager
    voice_manager = await get_voice_manager(voice_dir)
    logger.info(f"Voice manager initialized: {len(voice_manager.voices)} voices, {len(voice_manager.categories)} categories")

    # Create WebSocket server
    server = VoxCPMWebSocketServer(config.server, config.model)
    server.message_handler.set_voice_manager(voice_manager)

    # Create HTTP web server (for web UI)
    web_server = HTTPWebServer(
        host=config.server.host,
        port=config.server.port + 1,  # Web UI on port + 1 (9301)
        voice_manager=voice_manager
    )

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server, web_server)))

    # Start both servers
    try:
        # Start web server
        await web_server.start()
        logger.info(f"Web UI available at: {web_server.get_url()}")

        # Start WebSocket server (this blocks)
        await server.serve()
    except Exception as e:
        logger.exception(f"Server error: {e}")
    finally:
        logger.info("Server shutdown complete")


async def shutdown(server: VoxCPMWebSocketServer, web_server: HTTPWebServer = None):
    """Gracefully shutdown the servers."""
    logger.info("Shutting down...")
    try:
        await asyncio.wait_for(server.stop(), timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning("Server stop timed out")
    if web_server:
        try:
            await asyncio.wait_for(web_server.stop(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Web server stop timed out")


def run():
    """Run the server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")


if __name__ == "__main__":
    run()
