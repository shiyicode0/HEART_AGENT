# HEART_AGENT
# 🧸 儿童心理辅导语音陪伴机器人 (Voice-Interactive AI Companion for Kids)

![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-blue)
![Whisper](https://img.shields.io/badge/ASR-Whisper-green)
![Volcengine](https://img.shields.io/badge/TTS-Volcengine-red)

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
请确保你的电脑已安装 Python 3.8+ 环境。然后安装必要的依赖包：

```bash
pip install gradio requests numpy openai langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
```

###2. 配置密钥 (API Keys)
在运行代码前，请打开 app.py，找到代码顶部的 密钥配置 区域，填入你个人的 API Keys：

```bash
# 1. DeepSeek API Key
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 2. OpenAI Whisper API Key (支持官方或第三方中转 Key)
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
WHISPER_BASE_URL = "[https://api.your-proxy-domain.com/v1](https://api.your-proxy-domain.com/v1)" # 如果使用官方Key，可注释此行

# 3. 火山引擎 TTS 配置 (需开通语音合成服务)
VOLC_APPID = "xxxxxxxxxx"                                 
VOLC_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

###3. 运行程序
在终端中执行以下命令：

```Bash
python app.py
```
终端将会打印启动信息，并自动在浏览器中打开 http://127.0.0.1:8888。
