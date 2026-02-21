# MP3 HTTP Server

一个支持 AI 智能搜索的 MP3 音乐文件服务器。

## 功能特性

- 静态文件服务：直接通过 HTTP 访问 MP3 文件
- **混合智能搜索**：传统关键字匹配 + AI 语义理解
  - 单个匹配结果：直接返回，快速响应
  - 多个或无匹配结果：调用 AI 进行智能搜索
- 多线程支持：并发处理多个请求，互不干扰
- RESTful API：简洁的 JSON API 设计

## 快速启动

### 方式1：使用启动脚本（推荐）

```bash
./start_server.sh
```

### 方式2：直接运行 Python

```bash
# 设置环境变量（可选）
export OPENAI_API_BASE="http://192.168.13.228:8000/v1/"
export OPENAI_API_KEY="123"
export OPENAI_MODEL="Qwen3-30B-A3B"

# 启动服务器
python3 server.py
```

## API 接口

### 1. 列出所有歌曲

```bash
GET /api/list
GET /api/list?q=关键词  # 过滤列表
```

响应示例：
```json
{
  "total": 100,
  "files": [
    {
      "id": 1,
      "name": "七里香",
      "size": 5242880
    }
  ]
}
```

### 2. 搜索歌曲（混合智能搜索）

```bash
GET /api/search?q=七里香
GET /api/search?q=想听一首关于七里香的歌
GET /api/search?q=八度空间里的半兽人
```

特点：
- **智能混合搜索**：先使用传统关键字匹配
  - 如果只匹配到 1 首歌曲，直接返回（快速响应）
  - 如果匹配到多首或没有匹配，调用 AI 进行语义搜索
- 支持模糊搜索、语义理解、别名识别
- 自动优化性能：减少 AI 调用次数

示例查询：
- `q=七里香` - 精确匹配，直接返回
- `q=想听一首关于七里香的歌` - AI 语义理解
- `q=八度空间里的半兽人` - AI 理解意图
- `q=七里` - AI 模糊匹配

### 4. 随机获取歌曲

```bash
GET /api/random
GET /api/random?q=关键词  # 从匹配结果中随机
```

响应示例：
```json
{
  "id": 42,
  "name": "七里香",
  "size": 5242880
}
```

### 5. 下载歌曲

```bash
GET /api/download/{id}
```

参数：
- `id`: 歌曲 ID（从搜索接口获取）

## 配置

### AI 搜索配置

创建 `.env` 文件或设置环境变量：

```bash
OPENAI_API_BASE=http://192.168.13.228:8000/v1/
OPENAI_API_KEY=123
OPENAI_MODEL=Qwen3-30B-A3B
```

### 端口配置

默认端口：9100

修改端口：
```bash
export PORT=9101
python3 server.py
```

## 文件结构

```
music/
├── server.py          # 主服务器程序
├── start_server.sh    # 启动脚本
├── .env.example       # 配置示例文件
├── 周杰伦/            # 歌曲目录
└── dj/               # 歌曲目录
```

## 技术细节

### 混合智能搜索工作原理

1. **第一步：传统关键字匹配**
   - 对所有歌曲进行传统关键字匹配
   - 如果只匹配到 1 首歌曲，直接返回（快速响应）
   
2. **第二步：AI 语义搜索（条件触发）**
   - 如果匹配到多首歌曲：使用 AI 进行智能排序
   - 如果没有匹配歌曲：使用 AI 进行语义理解搜索
   - 收集歌曲列表（最多 500 首）
   - 发送用户查询和歌曲列表给大模型
   - 大模型分析查询意图并返回匹配的歌曲 ID
   - 服务器根据 ID 返回歌曲信息

3. **性能优化**
   - 减少不必要的 AI 调用
   - 精确匹配时直接返回，延迟更低
   - 复杂查询时使用 AI，保证准确性

### 多线程设计

- 使用 `ThreadingMixIn` 实现多线程
- 每个请求独立处理，互不阻塞
- 守护线程设置，主进程退出时自动清理

## 故障排查

### 搜索功能不工作

1. 检查服务器是否正常运行：
   ```bash
   curl http://192.168.1.169:9100/api/list
   ```

2. 检查 AI API 是否可访问：
   ```bash
   curl http://192.168.13.228:8000/v1/models
   ```

3. 检查日志输出中的 `[DEBUG]` 信息，查看搜索策略
   - `Traditional search found 1 matches` - 使用传统搜索
   - `Using AI search` - 使用 AI 搜索

### 端口被占用

修改端口：
```bash
export PORT=9200
python3 server.py
```

## 许可证

MIT
