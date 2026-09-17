# BabyCare — 基于 VLM 的婴儿安全监控

> Django 5 + YOLOv8 + Qwen3.5-4B VL（llama.cpp）+ 双声学模型（YAMNet / PANNs）+ ASR（Qwen3-ASR，llama.cpp）
>
> 视频线：每秒采样 1 帧 → YOLO 判画面里有无婴儿 → 每 10 秒选 3 帧交给 VLM 做语义级判定（口鼻遮盖、趴睡、五官被遮蔽等）→ 命中后落库并通过 Home Assistant 推送通知。
>
> 音频线（可选）：FFmpeg 从摄像头拉取 PCM → YAMNet + PANNs 双模型共识判哭声/人声 → 事件录音落盘 → ASR 模型转写描述。

## 作者说

项目思路：用 YOLO 识别画面里的人、宝宝等目标，**检测到了才把帧交给大模型**，识别内容由 prompt 配置，提示词完全自己写。建议两类用法：

- **描述类**：比如「图片为家庭摄像头部分画面，按时间命名，家里有一个爸爸、一个妈妈、一个宝宝，请综合三个画面描述下」。可以一天之后再用大模型做总结——宝宝这一天都做了什么；
- **警告类**：比如「宝宝是否趴睡了，回答是或否」。

通知目前依赖 Home Assistant（HA），没有部署 HA 就没有办法主动通知。

欢迎大家尝试、提意见。

说明几点：

- 本项目主要是 AI 写的，**建议让 AI 来部署和阅读**，代码质量一般；
- 项目起因是家里 GPU 显存一般都不大，跑不了大模型，所以用 4B 以下的小模型来实现，**理论上 6G 显存就能跑**，具体请自行尝试；
- 各模型默认是**分开单独跑**的，大模型自行启动，想换什么模型自己换就行；
- 显存不够的话，可以让 ASR 模型和 VL（视觉）模型**循环启停**（互斥占用）：VL 模型存活时间给长一些，并关闭思考模式。VL 处理单次请求比 ASR 慢很多很多，ASR 单请求 1 秒不到。

## 功能特性

- **视频接入**：本地视频文件（测试用） / ONVIF 摄像头（RTSP），1Hz 帧采样（FrameBus）
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
#
# 6.（可选）音频检测需另装独立 venv，见下节「音频线安装」，
#    不装的话音频功能不生效（主站不受影响）
```

## 音频线安装（可选；要用音频检测必须做，否则音频功能不生效）

音频检测（哭声/人声识别 + 录音 + ASR 转写）跑在**独立 venv** 里（依赖 torch /
tensorflow，与主环境隔离，避免版本冲突）。`.venv-audio/` 不入库，**每个部署都
要做一次**：

```powershell
# 1. 在项目根（manage.py 所在目录）创建独立虚拟环境
python -m venv .venv-audio

# 2. 安装音频线依赖（torch / tensorflow 体积较大，请耐心等待）
.venv-audio\Scripts\pip install -r requirements-audio.txt

# 3. 确认 .env 打开音频总开关（默认 false = 音频线永不启动）
#    BABYCARE_AUDIO_ENABLED=true

# 4. 确认 FFmpeg / FFprobe 在 PATH 上（或用 BABYCARE_FFMPEG_BIN 指定绝对路径）
ffmpeg -version
```

说明：

- 音频 worker 进程由网页「**音频控制**」页拉起/停止，不需要手动启动；
  web 侧默认用 `<项目根>/.venv-audio/Scripts/python.exe` 作为解释器
  （`.env` 里 `BABYCARE_AUDIO_PYTHON` 可覆盖路径）。
- 不装这个 venv（或路径不对）时：主站一切正常，但「音频控制」页会显示
  worker 启动失败/无心跳，哭声检测、事件录音、ASR 转写**全部不生效**。
- MySQL 用户注意：音频 venv 也要再装一次 `mysqlclient`（SQLite 无此问题）。
- 前置条件：摄像头需开启音频（ONVIF Profile 带 AudioEncoderConfiguration），
  事件描述功能还需另起 ASR llama-server（8136 端口，见「模型准备」）。

## 模型准备

模型资产分为两类：**YOLO / YAMNet / PANNs 类别表已随仓库提供**（`model/` 目录，共约 60 MB），**大模型 gguf 需自行下载**：

| 模型 | 用途 | 获取方式 |
|---|---|---|
| `best.pt`（自训练 YOLOv8） | 婴儿检测 | ✅ 随仓库提供 `model/best.pt` |
| `yolov8s.pt`（COCO 预训练） | 备用检测模型 | ✅ 随仓库提供 |
| YAMNet SavedModel | 哭声/人声检测 | ✅ 随仓库提供 `model/yamnet/` |
| PANNs 类别表 | 哭声/人声检测（第二意见） | ✅ 随仓库提供 `model/panns/class_labels_indices.csv`；权重 `Cnn14_mAP=0.431.pth`（约 300 MB）需自行下载到 `~\panns_data\`（`BABYCARE_AUDIO_PANNS_CHECKPOINT` 可覆盖） |
| Qwen3.5-4B `*.gguf` + `mmproj-*.gguf` | VLM 判定 | ❌ 自行下载，`BABYCARE_LLAMA_MODEL_DIR` 指定任意目录。量化档位通常 **Q4_K_M 就够用** |
| Qwen3-ASR-1.7B `*.gguf` | 音频转写（可选） | ❌ 自行下载，跑在 8136 端口。量化档位**能上 Q8 更好**（模型小，Q8 也占不了多少显存） |

两个 YOLO 文件的说明：

- **`model/best.pt`**：可识别**婴儿**，是作者单独训练的模型，也是本项目的主检测模型。
  如识别效果有问题，欢迎提 issue 反馈，作者会继续训练改进；
- **`model/yolov8s.pt`**：YOLOv8 官方预训练模型（COCO），可识别**人、猫**等多种类型，
  供扩展检测用途。

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
| 数据位置 | MySQL 服务端 | `data/db.sqlite3` |
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


