import os
import uuid
import numpy as np
import soundfile as sf
import pysrt
import torch
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from kokoro import KPipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from safetensors.torch import load_file
import librosa

# Hugging Face cache
HF_HOME = os.environ.get("HF_HOME", "/data/huggingface")
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

# TMP directory
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# Load Whisper
MODEL_DIR = "/data/.cache/whisper"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Whisper model not found at {MODEL_PATH}.")

print("🔹 Loading Whisper locally from:", MODEL_DIR)
processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
state_dict = load_file(MODEL_PATH, device="cpu")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    state_dict=state_dict,
    local_files_only=True,
)

# Kokoro TTS
_kpipeline = None
def get_pipeline(lang_code: str = "a"):
    global _kpipeline
    if _kpipeline is None:
        _kpipeline = KPipeline(lang_code=lang_code)
    return _kpipeline

# FastAPI
app = FastAPI(title="Kokoro-82M API", version="1.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=TMP_DIR), name="files")

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    lang_code: str = "a"
    speed: float = 1.0  # normal speed
    split_pattern: str = r"\n+"
    sample_rate: int = 24000

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/tts")
def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        pipeline = get_pipeline(req.lang_code)
        audio_parts = []
        for _, _, audio in pipeline(
            req.text,
            voice=req.voice,
            speed=req.speed,
            split_pattern=req.split_pattern
        ):
            audio_parts.append(audio)

        if not audio_parts:
            raise HTTPException(status_code=422, detail="No audio generated.")

        # Concatenate audio
        full_audio = np.concatenate(audio_parts)

        # Save WAV
        file_id = uuid.uuid4().hex
        wav_filename = f"tts_{file_id}.wav"
        wav_path = os.path.join(TMP_DIR, wav_filename)
        sf.write(wav_path, full_audio, req.sample_rate, format="WAV")
        duration = len(full_audio) / req.sample_rate

        # Resample to 16 kHz for Whisper
        speech_array, sr = sf.read(wav_path)
        speech_16k = librosa.resample(speech_array, orig_sr=sr, target_sr=16000)
        inputs = processor(speech_16k, sampling_rate=16000, return_tensors="pt")

        # Generate transcription with segments
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]

        # Simple word-level segmentation
        words = transcription.strip().split()
        total_words = len(words)
        word_durations = duration / max(total_words, 1)

        subs = pysrt.SubRipFile()
        current_time = 0
        for i, word in enumerate(words):
            start_ms = int(current_time * 1000)
            end_ms = int((current_time + word_durations) * 1000)
            subs.append(
                pysrt.SubRipItem(
                    index=i + 1,
                    start=pysrt.SubRipTime(milliseconds=start_ms),
                    end=pysrt.SubRipTime(milliseconds=end_ms),
                    text=word
                )
            )
            current_time += word_durations

        # Save SRT
        srt_filename = f"tts_{file_id}.srt"
        srt_path = os.path.join(TMP_DIR, srt_filename)
        subs.save(srt_path, encoding="utf-8")

        return {"wav": wav_path, "srt": srt_path, "duration": duration, "text": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/list-files")
def list_files():
    try:
        files = [f for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f))]
        return {"files": files}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)


@app.delete("/flash")
def delete_tmp_files():
    tmp_path = "/tmp"
    deleted = []

    try:
        for filename in os.listdir(tmp_path):
            file_path = os.path.join(tmp_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                deleted.append(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                deleted.append(file_path)
        return {"status": "success", "deleted": deleted}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
