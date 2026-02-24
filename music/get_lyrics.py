import requests
import re
import os
import json
from urllib.parse import quote


class LyricFetcher:
    BASE_URL = "https://music.163.com/api"
    LLM_API_URL = "http://localhost:9500/v1/chat/completions"
    LLM_API_KEY = "123"
    LLM_MODEL = "Qwen3-4B-Instruct-2507"

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://music.163.com",
        }

    def parse_filename_with_llm(self, filename):
        filename = filename.replace(".mp3", "")

        prompt = f"""请从以下音乐文件名中提取歌手和歌名信息。

文件名：{filename}

要求：
1. 只返回JSON格式，不要其他解释
2. 如果能识别出歌手和歌名，格式为：{{"artist": "歌手名", "title": "歌名"}}
3. 如果只能识别出歌名，格式为：{{"artist": null, "title": "歌名"}}
4. 歌名通常是歌曲的主体名称，不是版本信息（如DJ版、Remix版等）
5. 歌手是演唱者的名字

示例：
- "151.留什么给你小葡萄_DJ版" -> {{"artist": "留什么给你小葡萄", "title": "留什么给你"}}
- "267.自娱自乐金志文_Remix版" -> {{"artist": "金志文", "title": "自娱自乐"}}
- "075.爱情堡垒杨小壮_DJ版" -> {{"artist": "杨小壮", "title": "爱情堡垒"}}
"""

        try:
            response = requests.post(
                self.LLM_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.LLM_API_KEY}",
                },
                json={
                    "model": self.LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个音乐文件名解析专家，请从文件名中提取歌手和歌名信息，只返回JSON格式。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                },
                timeout=10,
            )

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            content_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if content_match:
                json_str = content_match.group(0)
                result = json.loads(json_str)
                return result.get("artist"), result.get("title")

            return None, filename

        except Exception as e:
            print(f"  LLM解析失败: {e}")
            return None, filename

    def search_song(self, keyword, limit=5):
        url = f"{self.BASE_URL}/search/get/web"
        params = {
            "csrf_token": "",
            "s": keyword,
            "type": "1",
            "offset": "0",
            "limit": str(limit),
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            data = response.json()

            if data.get("result") and data["result"].get("songs"):
                return data["result"]["songs"]
            return []
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def get_lyric(self, song_id):
        url = f"{self.BASE_URL}/song/lyric"
        params = {"id": str(song_id), "lv": "-1", "kv": "-1", "tv": "-1"}

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=10
            )
            data = response.json()

            if data.get("lrc"):
                return data["lrc"].get("lyric", "")
            return ""
        except Exception as e:
            print(f"获取歌词失败: {e}")
            return ""

    def get_lyric_for_file(self, filepath):
        filename = os.path.basename(filepath)
        artist, title = self.parse_filename_with_llm(filename)

        print(f"\n处理: {filename}")
        print(f"  歌手: {artist if artist else '未知'}")
        print(f"  歌名: {title}")

        if not artist:
            keyword = title
        else:
            keyword = f"{artist} {title}"

        print(f"  搜索关键词: {keyword}")

        songs = self.search_song(keyword)

        if not songs:
            print(f"  未找到匹配歌曲")
            return None

        for i, song in enumerate(songs[:3]):
            song_artists = ", ".join(
                [a.get("name", "") for a in song.get("artists", [])]
            )
            song_name = song.get("name", "")
            print(f"  匹配 {i + 1}: {song_name} - {song_artists}")

        best_song = songs[0]
        song_id = best_song.get("id")
        lyric = self.get_lyric(song_id)

        if lyric:
            print(f"  歌词获取成功")
            return lyric
        else:
            print(f"  未找到歌词")
            return None

    def save_lyric(self, lyric, filepath):
        lrc_path = filepath.replace(".mp3", ".lrc")

        try:
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lyric)
            print(f"  已保存: {lrc_path}")
            return True
        except Exception as e:
            print(f"  保存失败: {e}")
            return False


def main():
    fetcher = LyricFetcher()

    mp3_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))

    print(f"找到 {len(mp3_files)} 个MP3文件\n")

    success_count = 0
    fail_count = 0

    for mp3_file in mp3_files:
        lyric = fetcher.get_lyric_for_file(mp3_file)

        if lyric:
            if fetcher.save_lyric(lyric, mp3_file):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(mp3_files)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
