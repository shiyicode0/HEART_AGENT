# 🤖 AI 心理陪伴舱 (AI Voice Companion Pod)

一个面向 6-12 岁儿童的低延迟、全双工、Web 架构的语音心理陪伴机器人。
系统集成了本地知识库（RAG）、情感分析拦截、极速语音合成（TTS）与流式大模型交互，旨在为儿童提供安全、温柔、稳定的心理疏导与陪伴。

## 📁 目录结构

```text
my_project/
├── main.py              # 核心业务逻辑、WebSocket 引擎与 Web 服务入口
├── config.py            # 系统配置文件（API Keys、端口号等）
├── prompts.py           # 核心提示词（System Prompt）与 Few-Shots 对话示例
├── knowledge_test.txt   # 本地 RAG 语料库（首次运行自动生成）
└── README.md            # 项目说明文档
```

🚀 快速启动

1. 环境准备
请确保你的电脑已安装 Python 3.8+ 版本。然后在终端中运行以下命令安装所需依赖：
```Bash
pip install fastapi uvicorn websockets httpx openai langchain-community langchain-text-splitters sentence-transformers faiss-cpu
```

2. 配置密钥

```
打开 config.py 文件，填入你自己的 API 密钥：

DEEPSEEK_API_KEY: DeepSeek 大模型密钥

OPENAI_API_KEY: Whisper 语音识别密钥

VOLC_APPID & VOLC_TOKEN: 火山引擎 TTS 的应用 ID 和令牌
```

3. 自定义语料库 
你可以修改 knowledge_test.txt 中的内容，将你们学校或心理机构的特定信息写入其中。每次用户发问时，系统会进行相似度检索并融入对话上下文。

4. 运行服务
在终端中执行以下命令启动系统：
```Bash
python main.py
```
