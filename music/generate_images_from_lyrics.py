#!/usr/bin/env python3
"""
Generate images from song lyrics using LLM and Image Generation API
Multi-queue architecture with 2 threads for LLM processing and 1 thread for image generation
- If .lrc file exists, use lyrics content to generate prompt
- If .lrc file does not exist, use song name to generate prompt
- AI will automatically select the best artistic style based on song emotion and theme
"""

import os
import re
import json
import time
import threading
import argparse
from queue import Queue
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from tqdm import tqdm

load_dotenv()

# LLM Configuration
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:9500/v1/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "123")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-4B-Instruct-2507")

# Thread counts
LLM_THREAD_COUNT = 2
IMAGE_THREAD_COUNT = 1


class LyricsImageGenerator:
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)

        print("Loading Z-Image-Turbo model...")
        self.pipe = ZImagePipeline.from_pretrained(
            "Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.pipe.to("cuda")
        print("Model loaded successfully")

        self.success_count = 0
        self.fail_count = 0
        self.errors = []
        self.lock = threading.Lock()
        self.total_files = 0
        self.processed_count = 0
        self.pbar = None

        self.llm_queue = Queue()
        self.image_queue = Queue()
        self.stop_event = threading.Event()

    def extract_lyrics(self, lrc_path):
        """Extract lyrics content, filter out metadata"""
        try:
            with open(lrc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lyrics_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[00:"):
                    lyrics_lines.append(line)
                elif not re.match(r"\[.*?\]", line) and len(line) > 2:
                    lyrics_lines.append(line)

            lyrics_text = "\n".join(lyrics_lines)
            return lyrics_text
        except Exception as e:
            print(f"[LLM-{threading.current_thread().name}] Error reading lyrics: {e}")
            return ""

    def generate_prompt_with_llm(self, lyrics, song_name):
        """Generate image prompt using LLM"""
        if lyrics and len(lyrics.strip()) > 10:
            prompt = f"""你是一个专业的图片提示词生成专家，擅长将歌词转化为视觉化的艺术描述。

歌曲名称：{song_name}

歌词内容：
{lyrics[:2000]}

要求：
1. 深入分析歌词的情感基调（忧伤、快乐、思念、孤独、激昂、宁静、怀旧、梦幻等）
2. 提取歌词中的关键意象（雨、月光、离别、重逢、星空、城市、大海、山川、花海、夕阳等）
3. 根据歌曲的情感和主题，选择最合适的艺术风格，必须从以下风格中选择一种：
   - 实摄影风格：真实感强，注重光影和质感
   - 电影感风格：电影镜头感，画面叙事性强
   - 油画风格：厚重质感，色彩浓郁
   - 水彩风格：轻盈通透，晕染效果
   - 写实插画风格：细节丰富，介于写实与插画之间
   - 极简风格：简洁干净，留白设计
   - 超现实风格：梦幻朦胧，如梦似幻
   - 古典绘画风格：古典优雅，如油画水彩
4. 不要使用日系动漫、二次元、动漫人物、漫画风格
5. 生成一个详细的场景描述，包含：场景氛围、主要人物/物体、色调、光线、构图
6. 提示词应该直接用于AI图片生成，确保画面感强烈

只返回提示词，不要包含其他解释。"""
        else:
            prompt = f"""你是一个专业的图片提示词生成专家，擅长根据歌曲名称生成视觉化的艺术描述。

歌曲名称：{song_name}

要求：
1. 根据歌曲名称推测歌曲的主题和情感
2. 根据歌曲的情感和主题，选择最合适的艺术风格，必须从以下风格中选择一种：
   - 实摄影风格：真实感强，注重光影和质感
   - 电影感风格：电影镜头感，画面叙事性强
   - 油画风格：厚重质感，色彩浓郁
   - 水彩风格：轻盈通透，晕染效果
   - 写实插画风格：细节丰富，介于写实与插画之间
   - 极简风格：简洁干净，留白设计
   - 超现实风格：梦幻朦胧，如梦似幻
   - 古典绘画风格：古典优雅，如油画水彩
3. 不要使用日系动漫、二次元、动漫人物、漫画风格
4. 生成一个详细的场景描述，包含：场景氛围、主要人物/物体、色调、光线、构图
5. 提示词应该直接用于AI图片生成，确保画面感强烈

只返回提示词，不要包含其他解释。"""

        try:
            request_data = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的图片提示词生成专家，擅长将歌词转化为视觉化的艺术描述。你必须从实摄影、电影感、油画、水彩、写实插画、极简、超现实、古典绘画等风格中选择最合适的风格，绝对不能使用日系动漫、二次元或漫画风格。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 500,
            }

            response = self.llm_client.chat.completions.create(**request_data)
            generated_prompt = response.choices[0].message.content.strip()
            return generated_prompt

        except Exception as e:
            print(f"[LLM-{threading.current_thread().name}] Error calling LLM: {e}")
            return None

    def save_prompt(self, prompt, save_path):
        """Save prompt to -imggen.txt file"""
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"[LLM-{threading.current_thread().name}] Saved prompt: {save_path}")
            return True
        except Exception as e:
            print(f"[LLM-{threading.current_thread().name}] Error saving prompt: {e}")
            return False

    def generate_image(self, prompt, song_name, mp3_path):
        """Generate image using local Z-Image-Turbo model"""
        try:
            generated_seed = int(time.time())
            mp3_dir = Path(mp3_path).parent
            filename = f"{song_name}.png"
            local_path = mp3_dir / filename

            print(
                f"[IMG-{threading.current_thread().name}] Generating image: {filename}"
            )

            image = self.pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=torch.Generator("cuda").manual_seed(generated_seed % (2**32)),
            ).images[0]

            img_resized = image.resize((240, 240), Image.Resampling.LANCZOS)
            img_resized.save(local_path, "PNG")

            print(f"[IMG-{threading.current_thread().name}] Image saved: {local_path}")
            return str(local_path)

        except Exception as e:
            import traceback

            error_msg = str(e) if e else "Unknown error"
            traceback_str = traceback.format_exc()
            print(
                f"[IMG-{threading.current_thread().name}] Error generating image: {error_msg}"
            )
            print(f"[IMG-{threading.current_thread().name}] Traceback: {traceback_str}")
            return None

    def llm_worker(self, worker_id):
        """LLM worker thread"""
        thread_name = f"LLM-{worker_id}"
        threading.current_thread().name = thread_name

        while not self.stop_event.is_set():
            try:
                task = self.llm_queue.get(timeout=1)
                if task is None:
                    break

                mp3_path = task["mp3_path"]
                song_name = task["song_name"]

                if self.pbar:
                    self.pbar.set_description(f"[{thread_name}] {song_name[:30]}")

                lrc_path = mp3_path.replace(".mp3", ".lrc")
                lyrics = ""

                if os.path.exists(lrc_path):
                    lyrics = self.extract_lyrics(lrc_path)

                prompt = self.generate_prompt_with_llm(lyrics, song_name)
                if prompt:
                    save_path = mp3_path + "-imggen.txt"
                    self.save_prompt(prompt, save_path)

                    self.image_queue.put(
                        {"song_name": song_name, "prompt": prompt, "mp3_path": mp3_path}
                    )
                else:
                    with self.lock:
                        self.fail_count += 1
                        self.errors.append(
                            {"song": song_name, "error": "LLM prompt generation failed"}
                        )
                        if self.pbar:
                            self.pbar.write(
                                f"✗ [{thread_name}] Failed to generate prompt for {song_name}"
                            )

                self.llm_queue.task_done()

            except Exception as e:
                if not self.stop_event.is_set():
                    if self.pbar:
                        self.pbar.write(f"✗ [{thread_name}] Error: {e}")

    def image_worker(self, worker_id):
        """Image generation worker thread"""
        thread_name = f"IMG-{worker_id}"
        threading.current_thread().name = thread_name

        while not self.stop_event.is_set():
            try:
                task = self.image_queue.get(timeout=60)
                if task is None:
                    break

                song_name = task["song_name"]
                prompt = task["prompt"]
                mp3_path = task["mp3_path"]

                if self.pbar:
                    self.pbar.set_description(
                        f"[{thread_name}] Generating: {song_name[:25]}"
                    )

                image_path = self.generate_image(prompt, song_name, mp3_path)
                if image_path:
                    with self.lock:
                        self.success_count += 1
                        if self.pbar:
                            self.pbar.update(1)
                            self.pbar.write(
                                f"✓ [{thread_name}] Image saved: {Path(image_path).name}"
                            )
                else:
                    with self.lock:
                        self.fail_count += 1
                        self.errors.append(
                            {"song": song_name, "error": "Image generation failed"}
                        )
                        if self.pbar:
                            self.pbar.write(
                                f"✗ [{thread_name}] Image generation failed for {song_name}"
                            )

                self.image_queue.task_done()

            except Exception as e:
                if not self.stop_event.is_set():
                    if self.pbar:
                        self.pbar.write(f"✗ [{thread_name}] Error: {e}")

    def process_all(self):
        """Process all MP3 files with multi-threaded queues"""
        mp3_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.lower().endswith(".mp3"):
                    mp3_files.append(os.path.join(root, file))

        self.total_files = len(mp3_files)
        print(f"\nFound {self.total_files} MP3 files")
        print(f"Starting LLM workers: {LLM_THREAD_COUNT} threads")
        print(f"Starting Image workers: {IMAGE_THREAD_COUNT} threads")
        print(f"{'=' * 60}\n")

        self.pbar = tqdm(
            total=self.total_files,
            desc="Processing",
            unit="song",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for mp3_file in mp3_files:
            self.llm_queue.put({"mp3_path": mp3_file, "song_name": Path(mp3_file).stem})

        llm_threads = []
        for i in range(LLM_THREAD_COUNT):
            t = threading.Thread(target=self.llm_worker, args=(i + 1,), daemon=True)
            t.start()
            llm_threads.append(t)

        image_threads = []
        for i in range(IMAGE_THREAD_COUNT):
            t = threading.Thread(target=self.image_worker, args=(i + 1,), daemon=True)
            t.start()
            image_threads.append(t)

        self.llm_queue.join()

        for _ in range(LLM_THREAD_COUNT):
            self.llm_queue.put(None)

        self.image_queue.join()

        for _ in range(IMAGE_THREAD_COUNT):
            self.image_queue.put(None)

        self.stop_event.set()

        for t in llm_threads:
            t.join(timeout=1)
        for t in image_threads:
            t.join(timeout=1)

        self.pbar.close()
        self.print_summary()

    def process_single(self, mp3_path):
        """Process a single MP3 file"""
        if not os.path.exists(mp3_path):
            print(f"Error: File not found: {mp3_path}")
            return

        song_name = Path(mp3_path).stem
        print(f"\n{'=' * 60}")
        print(f"Processing single file: {song_name}")
        print(f"{'=' * 60}")

        lrc_path = mp3_path.replace(".mp3", ".lrc")
        lyrics = ""

        if os.path.exists(lrc_path):
            lyrics = self.extract_lyrics(lrc_path)
            if lyrics:
                print(f"Extracted {len(lyrics)} characters of lyrics")
            else:
                print("No lyrics extracted, will use song name")
        else:
            print("No lyrics file found, will use song name")

        prompt = self.generate_prompt_with_llm(lyrics, song_name)
        if not prompt:
            print("Failed to generate prompt")
            return

        print(f"Generated prompt: {prompt[:80]}...")

        save_path = mp3_path + "-imggen.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Saved prompt: {save_path}")

        print("\nGenerating image...")
        image_path = self.generate_image(prompt, song_name, mp3_path)
        if image_path:
            print(f"✓ Success! Image saved: {image_path}")
        else:
            print("✗ Failed to generate image")

        print(f"{'=' * 60}\n")

    def print_summary(self):
        """Print processing summary"""
        print(f"\n{'=' * 60}")
        print(f"Processing Complete!")
        print(f"{'=' * 60}")
        print(f"Success: {self.success_count}")
        print(f"Failed:  {self.fail_count}")
        print(f"Total:   {self.success_count + self.fail_count}")

        if self.errors:
            print(f"\nFailed songs:")
            for error in self.errors[:10]:
                print(f"  - {error['song']}: {error['error']}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from song lyrics using LLM and Image Generation"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a single MP3 file (test mode)",
    )
    args = parser.parse_args()

    generator = LyricsImageGenerator()

    if args.file:
        generator.process_single(args.file)
    else:
        generator.process_all()


if __name__ == "__main__":
    main()
