# ==========================================
# main.py - 核心业务逻辑与 Web 服务入口 (满血知识库版)
# ==========================================
import os
import io
import uuid
import asyncio
import httpx
import webbrowser
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from openai import AsyncOpenAI

# 导入知识库相关组件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 导入拆分出去的配置和提示词模块
import config
import prompts

# ================= 1. 初始化客户端与模型 =================
# 解决 HuggingFace 国内下载网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

llm_client = AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
whisper_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.WHISPER_BASE_URL)

# ================= 2. 初始化 RAG 本地知识库 =================
def init_rag():
    print("📚 正在加载本地知识库...")
    file_path = "knowledge_test.txt"
    # 如果没有知识库文件，自动生成一个纯净通用的心理安抚库
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("【补充背景库】如果小朋友情绪非常激动或难过，可以温和地建议他们做三次深呼吸，或者去喝一杯温水放松一下。\n")
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10).split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ 知识库加载完毕！")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        print(f"⚠️ 知识库加载失败，降级为无记忆模式: {e}")
        return None

db = init_rag()

# ================= 3. 辅助函数 =================
def safety_filter(text):
    danger_words = ["不想活", "想消失", "死", "自杀"]
    for word in danger_words:
        if word in text:
            return False, "你现在的安全最重要。先别一个人撑着，请马上去找身边的大人。"
    return True, text

async def text_to_audio_async(text: str) -> str:
    if not text: return ""
    headers = {"Authorization": f"Bearer; {config.VOLC_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "app": {"appid": config.VOLC_APPID, "token": config.VOLC_TOKEN, "cluster": "volcano_tts"},
        "user": {"uid": "test_user"},
        "audio": {"voice_type": "zh_female_vv_uranus_bigtts", "encoding": "mp3", "speed_ratio": 1.0},
        "request": {"reqid": str(uuid.uuid4()), "text": text, "text_type": "plain", "operation": "query"}
    }
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post("https://openspeech.bytedance.com/api/v1/tts", json=payload, headers=headers, timeout=10.0)
            data = res.json()
            if data.get("code") == 3000:
                return data["data"]
        except Exception as e:
            print(f"TTS 报错: {e}")
    return ""

# ================= 4. FastAPI 与 前端 UI =================
app = FastAPI()

html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>心理陪伴舱</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .no-select { user-select: none; -webkit-user-select: none; }
    </style>
</head>
<body class="bg-gray-900 text-white h-screen flex items-center justify-center font-sans overflow-hidden">
    
    <div class="w-full max-w-md bg-gray-800 rounded-2xl shadow-2xl p-6 border border-gray-700 flex flex-col h-[95vh]">
        <h1 class="text-2xl font-bold text-center mb-2 text-indigo-400">🤖 心理陪伴舱</h1>
        
        <div id="status-indicator" class="text-center text-sm font-bold text-gray-500 mb-4 h-6 transition-colors">
            未连接
        </div>
        
        <div id="chat-box" class="flex-1 overflow-y-auto mb-4 p-4 bg-gray-900 rounded-lg border border-gray-700 space-y-3 text-sm flex flex-col">
            <div class="text-gray-400 text-center">请先连接服务器...</div>
        </div>

        <div class="space-y-3 relative">
            <button id="btn-turn" class="no-select w-full bg-blue-600 text-white font-bold py-4 rounded-xl transition cursor-not-allowed opacity-50" disabled>
                🎙️ 按住说话 松开发送
            </button>
            
            <button id="btn-stream" class="w-full bg-emerald-600 text-white font-bold py-4 rounded-xl transition cursor-not-allowed opacity-50" disabled
                    onclick="toggleAutoMode()">
                🌊 开启全自动免提
            </button>

            <button id="btn-connect" class="absolute top-0 left-0 w-full h-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-4 rounded-xl transition z-10"
                    onclick="connectWS()">
                🔗 点击初始化麦克风
            </button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const indicator = document.getElementById('status-indicator');
        const btnTurn = document.getElementById('btn-turn');
        const btnStream = document.getElementById('btn-stream');
        
        let ws;
        let globalStream;
        let mediaRecorder;
        let audioChunks = [];
        
        let isSystemBusy = false; 
        let isAutoMode = false;   
        let isRecording = false;  
        
        let audioQueue = [];
        let currentAudioPlayer = null;
        
        let audioContext, analyser, dataArray, silenceTimer;
        let checkSilenceFrameId;

        function updateStatus(text, colorClass) {
            indicator.innerText = text;
            indicator.className = `text-center text-sm font-bold mb-4 h-6 transition-colors ${colorClass}`;
        }

        function addLog(msg, role) {
            let color = role === 'user' ? 'text-green-400 self-end bg-gray-700 px-3 py-2 rounded-lg max-w-[85%]' : 
                        role === 'ai' ? 'text-blue-300 self-start bg-gray-800 border border-gray-700 px-3 py-2 rounded-lg max-w-[85%]' : 
                        'text-gray-400 text-center w-full text-xs';
            chatBox.innerHTML += `<div class="${color}">${msg}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function forceUnlock() {
            isSystemBusy = false;
            if (isAutoMode) {
                updateStatus("🟢 AI 讲完了，你可以说话了", "text-emerald-400");
                startAutoRecording(); 
            } else {
                updateStatus("✅ 就绪，请按住按钮说话", "text-blue-400");
            }
        }

        function interruptAI() {
            audioQueue = []; 
            if (currentAudioPlayer) {
                currentAudioPlayer.pause();
                currentAudioPlayer = null;
            }
            isSystemBusy = false;
        }

        function playNextAudio() {
            if (audioQueue.length === 0) {
                forceUnlock();
                return;
            }
            
            isSystemBusy = true;
            updateStatus("🔊 AI 正在说话...", "text-blue-400");
            
            currentAudioPlayer = new Audio("data:audio/mp3;base64," + audioQueue.shift());
            currentAudioPlayer.onended = playNextAudio;
            currentAudioPlayer.play();
        }

        async function connectWS() {
            try {
                updateStatus("正在请求麦克风...", "text-yellow-400");
                globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
                
                ws.onopen = () => {
                    updateStatus("✅ 连接成功，准备就绪", "text-blue-400");
                    document.getElementById('btn-connect').style.display = 'none';
                    btnTurn.disabled = false;
                    btnTurn.classList.remove('cursor-not-allowed', 'opacity-50');
                    btnTurn.classList.add('hover:bg-blue-500', 'active:bg-blue-700');
                    btnStream.disabled = false;
                    btnStream.classList.remove('cursor-not-allowed', 'opacity-50');
                    btnStream.classList.add('hover:bg-emerald-500');
                    
                    initAudioAnalyzer();
                };

                ws.onmessage = (event) => {
                    let data = JSON.parse(event.data);
                    
                    if (data.type === "status") {
                        if (data.state === "processing") {
                            isSystemBusy = true; 
                            updateStatus("🧠 AI 正在思考中...", "text-yellow-400");
                        } else if (data.state === "finished") {
                            forceUnlock();
                        }
                    } else if (data.type === "error") {
                        addLog(`❌ 系统异常: ${data.text}`, "sys");
                        forceUnlock(); 
                    } else if (data.type === "transcription") {
                        addLog(`${data.text}`, "user");
                    } else if (data.type === "ai_text") {
                        addLog(`${data.text}`, "ai");
                    } else if (data.type === "audio_chunk") {
                        audioQueue.push(data.audio);
                        if (!currentAudioPlayer || currentAudioPlayer.paused) {
                            playNextAudio();
                        }
                    }
                };
                
                ws.onclose = () => {
                    updateStatus("🔴 与服务器断开连接", "text-red-500");
                    forceUnlock();
                }
            } catch (err) {
                updateStatus("❌ 麦克风权限被拒绝", "text-red-500");
                alert("必须允许麦克风权限才能使用！");
            }
        }

        function initAudioAnalyzer() {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(globalStream);
            source.connect(analyser);
            analyser.fftSize = 256;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
        }

        function startManualRecording(e) {
            if (e) e.preventDefault();
            if (isAutoMode) return; 
            
            interruptAI(); 
            
            btnTurn.classList.replace('bg-blue-600', 'bg-red-600');
            btnTurn.innerText = "🔴 正在录音... 松开发送";
            updateStatus("🔴 录音中...", "text-red-400");
            
            audioChunks = [];
            mediaRecorder = new MediaRecorder(globalStream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.start();
            isRecording = true;
        }

        function stopManualRecording(e) {
            if (e) e.preventDefault();
            if (!isRecording || isAutoMode) return;
            
            btnTurn.classList.replace('bg-red-600', 'bg-blue-600');
            btnTurn.innerText = "🎙️ 按住说话 松开发送";
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                if (ws && audioBlob.size > 0) ws.send(audioBlob);
                isSystemBusy = true; 
                updateStatus("🚀 上传中...", "text-yellow-400");
            };
            mediaRecorder.stop();
            isRecording = false;
        }

        btnTurn.addEventListener('mousedown', startManualRecording);
        btnTurn.addEventListener('touchstart', startManualRecording);
        btnTurn.addEventListener('mouseup', stopManualRecording);
        btnTurn.addEventListener('touchend', stopManualRecording);
        btnTurn.addEventListener('mouseleave', stopManualRecording); 

        function toggleAutoMode() {
            isAutoMode = !isAutoMode;
            if (isAutoMode) {
                interruptAI();
                btnStream.innerText = "🛑 关闭自动免提";
                btnStream.classList.replace('bg-emerald-600', 'bg-red-600');
                btnTurn.disabled = true; 
                btnTurn.classList.add('opacity-50');
                
                addLog("🌊 已进入流式通话。停顿 0.8 秒自动发送。", "sys");
                startAutoRecording();
                checkSilence(); 
            } else {
                btnStream.innerText = "🌊 开启全自动免提";
                btnStream.classList.replace('bg-red-600', 'bg-emerald-600');
                btnTurn.disabled = false;
                btnTurn.classList.remove('opacity-50');
                
                clearTimeout(silenceTimer);
                cancelAnimationFrame(checkSilenceFrameId);
                
                if (isRecording) {
                    mediaRecorder.onstop = null; 
                    mediaRecorder.stop();
                    isRecording = false;
                }
                updateStatus("✅ 就绪，请选择模式", "text-blue-400");
                addLog("🛑 已退出流式模式", "sys");
            }
        }

        function startAutoRecording() {
            if (!isAutoMode || isSystemBusy || isRecording) return;
            
            audioChunks = [];
            mediaRecorder = new MediaRecorder(globalStream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                if (ws && audioBlob.size > 0 && isAutoMode) {
                    ws.send(audioBlob);
                }
            };
            
            mediaRecorder.start();
            isRecording = true;
        }

        function checkSilence() {
            if (!isAutoMode) return; 
            checkSilenceFrameId = requestAnimationFrame(checkSilence);

            if (isSystemBusy) {
                if (isRecording) {
                    mediaRecorder.onstop = null; 
                    mediaRecorder.stop();
                    isRecording = false;
                }
                return; 
            }

            if (!isRecording && !isSystemBusy) {
                startAutoRecording();
            }

            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for(let i=0; i<dataArray.length; i++) sum += dataArray[i];
            let volume = sum / dataArray.length;

            if (volume > 15) { 
                updateStatus("🟢 检测到声音...", "text-emerald-400");
                clearTimeout(silenceTimer);
                silenceTimer = setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.stop();
                        isRecording = false;
                        isSystemBusy = true; 
                        updateStatus("🚀 上传中...", "text-yellow-400");
                    }
                }, 800); 
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    return HTMLResponse(html_content)

# ================= 5. WebSocket 异步后端对讲 =================
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chat_history = []
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            
            if len(audio_bytes) < 1000: 
                await websocket.send_json({"type": "status", "state": "finished"})
                continue 
            
            await websocket.send_json({"type": "status", "state": "processing"})
            
            try:
                buf = io.BytesIO(audio_bytes)
                buf.name = "audio.webm"
                
                transcription = await whisper_client.audio.transcriptions.create(
                    model="whisper-1", file=buf, language="zh"
                )
                user_text = transcription.text.strip()
                
                if not user_text or len(user_text) < 2:
                    await websocket.send_json({"type": "status", "state": "finished"})
                    continue
                    
                await websocket.send_json({"type": "transcription", "text": user_text})
                
                is_safe, warning_msg = safety_filter(user_text)
                if not is_safe:
                    audio_b64 = await text_to_audio_async(warning_msg)
                    await websocket.send_json({"type": "ai_text", "text": warning_msg})
                    if audio_b64: await websocket.send_json({"type": "audio_chunk", "audio": audio_b64})
                    continue

                # 💡 核心：知识库检索 (RAG)
                context_info = ""
                if db:
                    ret = db.similarity_search(user_text, k=1)
                    if ret: 
                        context_info = f"\n\n【参考背景/语料库资料】：{ret[0].page_content}"

                # 把带有知识库的完整提示词喂给大模型
                messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT + context_info}] + prompts.FEW_SHOTS + chat_history
                messages.append({"role": "user", "content": user_text})

                response = await llm_client.chat.completions.create(
                    model="deepseek-chat", messages=messages, temperature=0.5, max_tokens=150, stream=True
                )
                
                ai_reply = ""
                current_sentence = ""
                punctuation = {'。', '！', '？', '；', '\n', '.', '!', '?', ';'}
                
                async for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta: continue
                    
                    ai_reply += delta
                    current_sentence += delta
                    
                    if any(p in delta for p in punctuation):
                        sentence = current_sentence.strip()
                        if sentence:
                            await websocket.send_json({"type": "ai_text", "text": sentence})
                            audio_b64 = await text_to_audio_async(sentence)
                            if audio_b64: await websocket.send_json({"type": "audio_chunk", "audio": audio_b64})
                            current_sentence = ""
                            
                if current_sentence.strip():
                    sentence = current_sentence.strip()
                    await websocket.send_json({"type": "ai_text", "text": sentence})
                    audio_b64 = await text_to_audio_async(sentence)
                    if audio_b64: await websocket.send_json({"type": "audio_chunk", "audio": audio_b64})
                        
                chat_history.append({"role": "user", "content": user_text})
                chat_history.append({"role": "assistant", "content": ai_reply})
                if len(chat_history) > 10: chat_history = chat_history[-10:]

            except Exception as inner_e:
                print(f"处理请求时报错: {inner_e}")
                await websocket.send_json({"type": "error", "text": "后台 API 网络波动，请重试"})
                await websocket.send_json({"type": "status", "state": "finished"})

    except WebSocketDisconnect:
        print("用户主动断开连接")

if __name__ == "__main__":
    print("🚀 携带语料库记忆的 Web 引擎启动中...")
    
    url = f"http://{config.HOST}:{config.PORT}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    print(f"👉 如果浏览器没有自动弹出，请手动访问: {url}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)