import torchaudio
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.ASR import EncoderDecoderASR
import tempfile
from fastapi.responses import JSONResponse
import pydantic
from pydub import AudioSegment
from pydantic import BaseModel
import tempfile
import os
import sys
sys.path.append('/usr/bin/ffmpeg')
import argparse
app = FastAPI()

tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts"
)
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder"
)
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

@app.get("/tts/")
async def tts(text: str):
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        filepath = temp_file.name

    # TTS 및 Vocoder 실행
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)

    # 음성 파일 저장
    torchaudio.save(filepath, waveforms.squeeze(1), 22050)

    # 생성된 파일 반환
    headers = {
        "Content-Disposition": f'attachment; filename="{os.path.basename(filepath)}"'
    }
    return FileResponse(
        filepath, media_type="audio/wav", headers=headers,
    )

@app.post("/stt/")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(await file.read())
        filepath = temp_file.name

    # 음성 파일을 텍스트로 변환
    try:
        transcription = asr_model.transcribe_file(filepath)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}



#score1 <- python3 inference_wav.py --wav score_input.m4a --lang en --label_type1 pron --label_type2 prosody --device cpu --dir_model ./model_ckpt/
#score2 <- python3 inference_wav.py --wav score_input.m4a --lang en --label_type1 pron --label_type2 articulation --device cpu --dir_model ./model_ckpt/
#최종 스코어 = score1+score2

import subprocess
from typing import List
from .inference_wav import inference_wav


class AudioFile(BaseModel):
    files: List[UploadFile]

@app.post("/infer/")
async def predict(files: AudioFile = File(...)):
    # 업로드된 파일 처리
    for file in files.files:
        # m4a 파일을 wav로 변환
        m4a_file = f"temp_{file.filename}"
        wav_file = f"temp_{os.path.splitext(file.filename)[0]}.wav"
        with open(m4a_file, "wb") as buffer:
            buffer.write(await file.read())
        subprocess.run(["ffmpeg", "-i", m4a_file, wav_file])
        os.remove(m4a_file)

        # 추론 함수 호출하고 결과 반환
        score = inference_wav(wav_file)
        os.remove(wav_file)
        cmd = f"python3 inference_wav.py --wav {wav_file} --lang en --label_type1 pron --label_type2 prosody --device cpu --dir_model ./model_ckpt/"
        score = subprocess.check_output(cmd, shell=True).decode().split(": ")[-1]
        
        return {"score": score}
    
# curl -X POST \
#   http://localhost:10010/infer \
#   -H 'Content-Type: multipart/form-data' \
#   -F 'files=@score_input.m4a'