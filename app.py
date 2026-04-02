import gradio as gr
import requests
import uuid
import base64
import os
import wave
import io
import numpy as np

# ================= 0. 环境与配置 =================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  

from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================= 1. 密钥配置 =================
#DeepSeek 配置
DEEPSEEK_API_KEY = "sk-5e086f0d09cb4f57a1c1a40e54170092"
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

#  OpenAI Whisper 配置 
OPENAI_API_KEY = "sk-40WwvNZ2wF1m65OB5WZdrQFNyHaPXwmI9XPbjkV8XtdfacNO" 
WHISPER_BASE_URL = "https://api.chatanywhere.tech/v1"
whisper_client = OpenAI(api_key=OPENAI_API_KEY, base_url=WHISPER_BASE_URL) 

# 火山 TTS 语音合成
VOLC_APPID = "3473137619"                                 
VOLC_TOKEN = "2sP5sW_T3EvS_TseRLr9wDJXUo3t8GH8"       

# 防网络拦截配置
NO_PROXY = {"http": None, "https": None}

# ================= 2. 专业心理陪伴 Prompt =================
SYSTEM_PROMPT = """
你是一名面向 6-12 岁儿童的语音心理陪伴机器人。
职责：接住情绪；帮孩子用简单的话说出感受；给一个很小、很简单的动作；在需要时，鼓励孩子找大人。
语音输出要求：句子短、用词简单、一次只说一个重点、听起来温柔稳定。绝对不要长段落和复杂分析。
回复顺序：1.先接住情绪 2.再说清感受 3.给个小建议 4.必要时找大人。
"""

DEVELOPER_PROMPT = """
一、输出目标：适合儿童听、适合 TTS 念。
二、语言规则：口语化、短句优先。绝对不要 markdown 列表。
三、依赖防护：如果孩子说“只有你陪我”等，要温和加入“我可以陪你说说，但真正能帮你的人是家长或老师，你值得让真人保护你。”
"""

STYLE_PROMPT = """
【常用表达】我在听；没关系，慢慢说；先不用急；你不用一个人扛。
【节奏】每轮只推进一点；孩子乱时更短；孩子沉默时不施压。
"""

FEW_SHOTS = [
    {"user": "今天同学不跟我玩。", "assistant": "听起来，你心里有点难受。你想先告诉我，是谁不跟你玩吗？不想说也没关系。"},
    {"user": "我明天考试，我好怕。", "assistant": "会怕很正常。我们先做一件小事。慢慢呼吸三下，好吗？"},
    {"user": "我不想活了。", "assistant": "你现在的安全最重要。先别一个人撑着。请马上去找身边的大人。"}
]

def safety_filter(text):
    danger_words = ["不想活", "想消失", "想伤害", "被打", "被威胁", "死", "自杀"]
    for word in danger_words:
        if word in text:
            return False, "你现在的安全最重要。先别一个人撑着。请马上去找身边的大人。"
    return True, text

# ================= 3. RAG 知识库 =================
def init_rag():
    file_path = "knowledge_test.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("【补充背景库】想找人倾诉，可以去西电附小的阳光心理小屋找李老师。\n")
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10).split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except: return None

db = init_rag()

# ================= 4. 语音接口 =================
def audio_to_text(audio_data):
    if audio_data is None: return ""
    try:
        sample_rate, data = audio_data

        if len(data) < 1000:
            return "（[提示] 录音太短啦，请多说几个字哦）"

        # 1. 简单的音频安全转换 
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float64:
            data = data.astype(np.float32)

        # 2. 强取单声道
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 3. 增强音量
        data = data - np.mean(data)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 0.9

        data = np.clip(data, -1.0, 1.0)
        pcm_data = (data * 32767.0).astype(np.int16)

        # 4. 生成临时 WAV 文件喂给 Whisper
        temp_wav_path = "temp_whisper_input.wav"
        with wave.open(temp_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate) # 完美保留原始采样率
            wav_file.writeframes(pcm_data.tobytes())

        # 5. 🎯 调用 ChatAnywhere 的 Whisper 接口
        with open(temp_wav_path, "rb") as audio_file:
            transcription = whisper_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="zh"  # 指定中文，识别极快
            )
        
        text = transcription.text
        if not text.strip():
            return "（[提示] Whisper 听到声音了，但没分辨出有效的说话内容）"
            
        return text

    except Exception as e:
        return f"（[崩溃] Whisper 识别失败: {str(e)}。请检查你的网络或额度）"

def text_to_audio(text):
    if not text: return None
    try:
        headers = {
            "Authorization": f"Bearer; {VOLC_TOKEN}", 
            "Content-Type": "application/json"
        }
        payload = {
            "app": {"appid": VOLC_APPID, "token": VOLC_TOKEN, "cluster": "volcano_tts"},
            "user": {"uid": "test_user_123"},
            "audio": {"voice_type": "zh_female_vv_uranus_bigtts", "encoding": "mp3", "speed_ratio": 1.0},
            "request": {"reqid": str(uuid.uuid4()), "text": text, "text_type": "plain", "operation": "query"}
        }
        # 绕过代理发送 TTS 请求
        res = requests.post(
            "https://openspeech.bytedance.com/api/v1/tts", 
            json=payload, 
            headers=headers,
            proxies=NO_PROXY
        ).json()
        
        if res.get("code") == 3000:
            path = f"reply_{uuid.uuid4().hex[:4]}.mp3"
            with open(path, "wb") as f: f.write(base64.b64decode(res["data"]))
            return path
    except: pass
    return None

# ================= 5. AI 对话引擎 =================
def chat_engine(audio_file, text_input, history):
    history = history or []
    
    clean_history = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            clean_history.append({"role": item["role"], "content": item["content"]})
        elif hasattr(item, "role") and hasattr(item, "content"):
            clean_history.append({"role": item.role, "content": item.content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            clean_history.append({"role": "user", "content": str(item[0])})
            clean_history.append({"role": "assistant", "content": str(item[1])})
    history = clean_history

    user_msg = audio_to_text(audio_file) if audio_file else text_input
    if not user_msg: return history, None

    if user_msg.startswith("（["):
        history.append({"role": "user", "content": "🎤 录制了一段语音"})
        history.append({"role": "assistant", "content": f"🔴 异常：{user_msg}"})
        return history, None

    is_safe, warning_msg = safety_filter(user_msg)
    if not is_safe:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": warning_msg})
        return history, text_to_audio(warning_msg)

    context_info = ""
    if db:
        ret = db.similarity_search(user_msg, k=1)
        if ret: context_info = f"\n【参考背景】：{ret[0].page_content}"

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n{DEVELOPER_PROMPT}\n{STYLE_PROMPT}{context_info}"}]
    for shot in FEW_SHOTS:
        messages.append({"role": "user", "content": shot["user"]})
        messages.append({"role": "assistant", "content": shot["assistant"]})
    
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
                
    messages.append({"role": "user", "content": user_msg})

    try:
        #  DeepSeek 大脑思考
        response = llm_client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.5, max_tokens=150
        )
        ai_reply = response.choices[0].message.content
        
        is_safe_out, final_reply = safety_filter(ai_reply)
        audio_reply = text_to_audio(final_reply)
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": final_reply})
        return history, audio_reply
    except Exception as e:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": f"🔴 大模型报错: {str(e)}"})
        return history, None

# ================= 6. 极简 UI =================
with gr.Blocks(title="儿童心理辅导 ") as demo:
    gr.Markdown("### 🧠 儿童心理辅导全链路测试 ")
    chatbot = gr.Chatbot(label="对话记录") 
    
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ 录音")
        text_in = gr.Textbox(placeholder="📝 打字...", scale=3)
        btn = gr.Button("发送", variant="primary")
        
    audio_out = gr.Audio(label="🔊 语音回复", autoplay=True)

    def trigger_chat(audio, text, hist):
        new_hist, new_audio = chat_engine(audio, text, hist)
        return new_hist, new_audio, None, ""

    btn.click(trigger_chat, [audio_in, text_in, chatbot], [chatbot, audio_out, audio_in, text_in])
    text_in.submit(trigger_chat, [audio_in, text_in, chatbot], [chatbot, audio_out, audio_in, text_in])

if __name__ == "__main__":
    print("====== 当前运行版本： Whisper 引擎 ======")
    print("🚀  http://127.0.0.1:8888")
    demo.launch(server_port=8888, inbrowser=True)