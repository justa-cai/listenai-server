#!/usr/bin/env python3
"""
Simple WebSocket client for testing the VoxCPM TTS Server.

This example demonstrates how to connect to the server and request TTS.
"""

import asyncio
import websockets
import json
import uuid
import struct
import argparse


def parse_binary_frame(data: bytes):
    """
    Parse a binary frame from the server.

    Returns:
        tuple: (metadata dict, audio bytes)
    """
    if len(data) < 8:
        raise ValueError("Frame too short")

    # Parse header
    magic = struct.unpack('>H', data[0:2])[0]
    if magic != 0xAA55:
        raise ValueError(f"Invalid magic number: 0x{magic:04X}")

    msg_type = data[2]
    metadata_length = struct.unpack('>I', data[4:8])[0]

    # Parse metadata JSON
    metadata_json = data[8:8 + metadata_length].decode('utf-8')
    metadata = json.loads(metadata_json)

    # Parse audio payload
    payload_offset = 8 + metadata_length + 4
    payload_length = struct.unpack('>I', data[8 + metadata_length:payload_offset])[0]
    audio_data = data[payload_offset:payload_offset + payload_length]

    return metadata, audio_data


async def tts_request(
    text: str,
    url: str = "ws://localhost:9300/tts",
    mode: str = "streaming",
    output_file: str = None
):
    """
    Make a TTS request to the server.

    Args:
        text: Text to synthesize
        url: WebSocket server URL
        mode: 'streaming' or 'non_streaming'
        output_file: Optional file to save audio (as raw PCM)
    """
    request_id = str(uuid.uuid4())
    audio_chunks = []

    print(f"Connecting to {url}...")
    print(f"Request ID: {request_id}")
    print(f"Text: {text}")
    print(f"Mode: {mode}")
    print("-" * 50)

    try:
        async with websockets.connect(url, close_timeout=10) as ws:
            # Send request
            request = {
                "type": "tts_request",
                "request_id": request_id,
                "params": {
                    "text": text,
                    "mode": mode
                }
            }
            await ws.send(json.dumps(request))
            print("Request sent")

            # Receive response
            async for message in ws:
                if isinstance(message, bytes):
                    # Binary audio data
                    metadata, audio = parse_binary_frame(message)
                    sequence = metadata.get('sequence', 0)
                    sample_rate = metadata.get('sample_rate', 24000)

                    print(f"Received audio chunk: sequence={sequence}, "
                          f"size={len(audio)} bytes, sample_rate={sample_rate}")

                    audio_chunks.append((metadata, audio))

                else:
                    # JSON control message
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "progress":
                        print(f"Progress: {msg.get('state')} "
                              f"({msg.get('progress', 0)*100:.1f}%) - "
                              f"{msg.get('message', '')}")

                    elif msg_type == "complete":
                        result = msg.get("result", {})
                        print("-" * 50)
                        print(f"Complete!")
                        print(f"Duration: {result.get('duration', 0):.2f}s")
                        print(f"Samples: {result.get('samples', 0)}")
                        print(f"Chunks: {result.get('chunks', 0)}")
                        break

                    elif msg_type == "error":
                        error = msg.get("error", {})
                        print("-" * 50)
                        print(f"Error: {error.get('code', '')}")
                        print(f"Message: {error.get('message', '')}")
                        details = error.get('details', {})
                        if details:
                            print(f"Details: {json.dumps(details, indent=2)}")
                        break

            # Save audio if requested
            if audio_chunks and output_file:
                all_audio = b''.join(audio for _, audio in audio_chunks)
                with open(output_file, 'wb') as f:
                    f.write(all_audio)
                print(f"\nAudio saved to: {output_file}")

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server")
    except Exception as e:
        print(f"Error: {e}")
        raise


async def cancel_request(url: str = "ws://localhost:9300/tts"):
    """Send a cancel request."""
    request_id = str(uuid.uuid4())

    async with websockets.connect(url, close_timeout=10) as ws:
        cancel_msg = {
            "type": "cancel",
            "request_id": request_id
        }
        await ws.send(json.dumps(cancel_msg))
        print(f"Cancel request sent for {request_id}")


async def ping_server(url: str = "ws://localhost:9300/tts"):
    """Ping the server."""
    import time

    async with websockets.connect(url, close_timeout=10) as ws:
        start = time.time()
        ping_msg = {
            "type": "ping",
            "timestamp": int(time.time())
        }
        await ws.send(json.dumps(ping_msg))

        response = await ws.recv()
        pong = json.loads(response)
        latency = (time.time() - start) * 1000

        print(f"Pong received in {latency:.1f}ms")
        print(f"Server time: {pong.get('server_time')}")


def main():
    parser = argparse.ArgumentParser(description="VoxCPM TTS Client")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--url", default="ws://localhost:9300/tts",
                        help="WebSocket server URL")
    parser.add_argument("--mode", choices=["streaming", "non_streaming"],
                        default="streaming", help="TTS mode")
    parser.add_argument("--output", "-o", help="Output file for audio (raw PCM)")
    parser.add_argument("--ping", action="store_true", help="Ping the server")
    parser.add_argument("--cancel", action="store_true", help="Send cancel request")

    args = parser.parse_args()

    if args.ping:
        asyncio.run(ping_server(args.url))
    elif args.cancel:
        asyncio.run(cancel_request(args.url))
    elif args.text:
        asyncio.run(tts_request(args.text, args.url, args.mode, args.output))
    else:
        # Read from stdin
        import sys
        text = sys.stdin.read().strip()
        if text:
            asyncio.run(tts_request(text, args.url, args.mode, args.output))
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
