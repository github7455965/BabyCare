# BabyCare — 基于 VLM 的婴儿安全监控

> Django 5 + YOLOv8 + Qwen3.5-4B VL（llama.cpp）+ 双声学模型（YAMNet / PANNs）+ ASR（Qwen3-ASR，llama.cpp）
>
> 视频线：每秒采样 1 帧 → YOLO 判画面里有无婴儿 → 每 10 秒选 3 帧交给 VLM 做语义级判定（口鼻遮盖、趴睡、五官被遮蔽等）→ 命中后落库并通过 Home Assistant 推送通知。
>
> 音频线（可选）：FFmpeg 从摄像头拉取 PCM → YAMNet + PANNs 双模型共识判哭声/人声 → 事件录音落盘 → ASR 模型转写描述。

## 功能特性

- **视频接入**：本地视频文件 / ONVIF 摄像头（RTSP），1Hz 帧采样（FrameBus）
- **YOLO 前置过滤**：轻量模型判断画面有无婴儿，无婴儿不触发 VLM，省显存
- **VLM 语义检查**：Qwen3.5-4B 视觉模型按可配置的检查项（prompt）判定风险
- **检查项 / 摄像头 / 通知目标配置面板**：网页 CRUD，保存即生效
- **事件系统**：事件列表 / 详情（缩略图 + 三帧大图）/ 类型与时间筛选 / 分页
- **HA 通知**：手机 App 推送 + 音箱播报，多目标勾选，重复事件去重
- **音频检测**（可选）：哭声 / 人声双模型共识，事件录音 + ASR 转写
- **两种部署形态**：单机（YOLO/VLM 同机互斥）与分体（服务机 + 推理机分离）

## 环境要求

| 组件 | 说明 |
|---|---|
| Python | 3.11+（开发环境为 3.12） |
| 操作系统 | Windows / Linux 均可（音频线在 Windows 上开发验证） |
| FFmpeg / FFprobe | 音频线需要（PATH 上有即可，或用 `.env` 指定路径） |
| llama-server | VLM / ASR 推理需要（llama.cpp，单独进程运行） |
| GPU | 可选，见下文「显存需求」；无 GPU 时网页与配置功能不受影响 |

## 安装

```powershell
# 1. 安装依赖（默认 SQLite 数据库，无需任何数据库驱动）
pip install -r requirements.txt

# 2. 配置环境变量
copy .env.example .env
# 至少确认：DJANGO_SECRET_KEY、DB_ENGINE
# 上手推荐 DB_ENGINE=sqlite，其余项保持默认即可

# 3. 建表
python manage.py migrate

# 4. 启动
python -m daphne -b 127.0.0.1 -p 8124 config.asgi:application

# 5. 访问
#    http://127.0.0.1:8124/                  首页
#    http://127.0.0.1:8124/config/prompts/   VLM 检查项配置
#    http://127.0.0.1:8124/config/cameras/   摄像头配置
```

音频线运行在**独立 venv**（依赖 torch / tensorflow，与主环境隔离）：

```powershell
python -m venv .venv-audio
.venv-audio\Scripts\pip install -r requirements-audio.txt
# 音频 worker 由网页「音频控制」页拉起，解释器路径可用 BABYCARE_AUDIO_PYTHON 覆盖
```

## 模型准备

模型资产分为两类：**YOLO / YAMNet / PANNs 类别表已随仓库提供**（`model/` 目录，共约 60 MB），**大模型 gguf 需自行下载**：

| 模型 | 用途 | 获取方式 |
|---|---|---|
| `best.pt`（自训练 YOLOv8） | 婴儿检测 | ✅ 随仓库提供 `model/best.pt` |
| `yolov8s.pt`（COCO 预训练） | 备用检测模型 | ✅ 随仓库提供 |
| YAMNet SavedModel | 哭声/人声检测 | ✅ 随仓库提供 `model/yamnet/` |
| PANNs 类别表 | 哭声/人声检测（第二意见） | ✅ 随仓库提供 `model/panns/class_labels_indices.csv`；权重 `Cnn14_mAP=0.431.pth`（约 300 MB）需自行下载到 `~\panns_data\`（`BABYCARE_AUDIO_PANNS_CHECKPOINT` 可覆盖） |
| Qwen3.5-4B `*.gguf` + `mmproj-*.gguf` | VLM 判定 | ❌ 自行下载，`BABYCARE_LLAMA_MODEL_DIR` 指定任意目录 |
| Qwen3-ASR-1.7B `*.gguf` | 音频转写（可选） | ❌ 自行下载，跑在 8136 端口 |

llama-server 启动示例（VLM，端口 8082）：

```powershell
llama-server.exe ^
  -m <MODELS>\qwen35-4b-Q4_K_M.gguf ^
  --mmproj <MODELS>\mmproj-Qwen3.5-4B-F16.gguf ^
  --ctx-size 8192 ^
  -ngl 99 ^
  -ctk q8_0 -ctv q8_0 ^
  --cache-ram 0 ^
  -np 1 ^
  --host 127.0.0.1 ^
  --port 8082
```

> `--cache-ram 0` 与 `-np 1` 建议保留：llama.cpp 的 prompt cache 默认吃 8 GiB 宿主内存且只增不减；多 slot 的 KV cache 会白占大量显存。

## 显存需求

| 模式 | 变量 | 显存 | 说明 |
|---|---|---|---|
| 互斥（默认） | `BABYCARE_GPU_MODE=exclusive` | **6 GB** | YOLO 与 VLM 不同时常驻，任务驱动自动换入换出 |
| 并行 | `BABYCARE_GPU_MODE=parallel` | **22 GB** | YOLO + VLM 常驻并行，低延迟 |
| 分体部署 | `BABYCARE_GPU_MODE=off` | 按推理机配置 | 显存仲裁只在同机有意义 |

参考占用：VLM（Qwen3.5-4B Q4_K_M 权重约 2.2 GB + q8_0 KV cache，ctx 8192）合计约 3.5~4 GB；YOLO 小模型 < 0.5 GB；音频线 ASR（Qwen3-ASR-1.7B Q8_0）约 2.5 GB。

## 运行模式

**单机 vs 分体**（`BABYCARE_LLAMA_EXTERNAL`）：

| | 单机（默认） | 分体部署 |
|---|---|---|
| 服务 + 大模型 | 同一台机器 | 服务机 / 推理机各一台 |
| `BABYCARE_LLAMA_EXTERNAL` | `0` | **`1`**（必开） |
| `BABYCARE_LLAMA_SERVER_URL` | `http://127.0.0.1:8082` | `http://<推理机IP>:8082` |
| `BABYCARE_GPU_MODE` | `exclusive` / `parallel` | `off` |
| 服务进程能启停大模型吗 | 能（spawn / kill） | **不能**，只探活 |
| 大模型不可达时 | 报错 | 请求进数据库队列，恢复后自动回放（熔断器保护） |

分体模式必须开 `BABYCARE_LLAMA_EXTERNAL=1`，否则服务进程可能对远端 llama-server 执行卸载操作，把推理机上正在使用的服务关掉。

**循环 vs 常驻**（`BABYCARE_LLM_RESIDENT`，与上面正交，可任意组合）：

- `false`（默认）：每次 VLM 任务启停一次 llama-server，最省显存；
- `true`：YOLO/LLM 启动后常驻，按 `BABYCARE_LLM_AUTO_RESTART_HOURS`（默认 24h，`0` = 仅手动）周期重启防止显存膨胀；重启窗口内的请求进 `VLMQueuedTask` 队列，后台线程回放。

## 数据库：MySQL 还是 SQLite

由 `DB_ENGINE` 切换，两种都是完整支持：

| | `DB_ENGINE=mysql`（默认） | `DB_ENGINE=sqlite` |
|---|---|---|
| 适用 | 已有 MySQL / 多机 / 高并发写 | **单机部署，不想装数据库** |
| 依赖 | `pip install mysqlclient` | 无（Python 自带 `sqlite3`） |
| 数据位置 | MySQL 服务端 | `data/db.sqlite3`（已被 .gitignore 忽略） |
| 建库 | 需手动建 `babycare_vlm` | 自动创建 |
| `manage.py test` | 需要建库权限 | 开箱即用 |
| 跨机器共享 | 支持 | **不支持**（SQLite 不能放网络盘） |

SQLite 模式下 `settings.py` 会强制 `transaction_mode=IMMEDIATE` + WAL + `busy_timeout`：Django 在 SQLite 上会静默忽略 `select_for_update()`，音频 worker 的互斥只能靠「事务一开始就持有写锁」保证，不要绕过 settings.py 自行修改。

> MySQL 用户注意：音频线跑在独立 venv，`mysqlclient` 需要在主 venv 和音频 venv 各装一次。

## 项目结构

```
web_vlm_manage/
├── apps/
│   ├── streaming/     # 摄像头接入 + FrameBus（文件源 / ONVIF 源）
│   ├── yolo_detect/   # 1Hz 采样 + YOLO 检测 + 显存仲裁
│   ├── vlm/           # llama-server 客户端 / 进程管理 / 帧选择与落盘
│   ├── core/          # HTTP API（启停 / dismiss）+ 熔断器
│   ├── config_panel/  # 检查项 / 摄像头 / 通知目标配置面板
│   ├── dashboard/     # 首页 / 事件列表详情 / 控制页
│   ├── audio_detect/  # 音频线（独立进程 audio_worker）
│   └── cleanup/       # 数据清理管理命令
├── config/            # Django settings / urls / asgi / wsgi
├── home/              # 首页视图
├── templates/         # 全局模板
├── wsdl/              # ONVIF wsdl（onvif-zeep 不自带，需在此目录或用 .env 指定）
├── model/             # 模型权重（不入库，见「模型准备」）
├── media/             # 运行产物：帧图 / 事件录音（不入库）
└── data/              # 运行产物：SQLite 库 / pid / 日志（不入库）
```

## 测试

```powershell
python manage.py test
```

## 隐私说明

- `media/`（帧图、事件录音）与 `data/`（数据库、日志）都是运行时产物，已在 `.gitignore` 中排除，不会被提交；如果你的部署涉及真实婴儿画面/录音，请勿将这两个目录的任何内容发布到公共渠道。
- `.env` 存放密钥（数据库密码、ONVIF 密码、HA token 等），同样被 `.gitignore` 排除；请勿提交，也不要在 issue / 截图中泄露。
- 仓库内的 `.env.example` 只含占位值。
