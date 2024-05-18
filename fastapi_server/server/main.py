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
from .inference_wav import inference_wav
from .inference_wav import inference_wav2
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



#python3 inference_wav.py --wav score_input.m4a --lang en --label_type1 pron --label_type2 prosody --device cpu --dir_model ../model_ckpt/


class InferenceParams(BaseModel):
    lang: str = "en"
    label_type1: str = "pron"
    label_type2: str = "articulation"

class InferenceResult(BaseModel):
    total_score: float
    articulation_score: float
    prosody_score: float

@app.post("/infer/", response_model=InferenceResult)
async def scoring_audio(file: UploadFile = File(...), params: InferenceParams = None):
    if params is None:
        params = InferenceParams()

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_file:
        temp_file.write(await file.read())
        m4a_filepath = temp_file.name

    wav_filepath = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        track = AudioSegment.from_file(m4a_filepath, format="m4a")
        track.export(wav_filepath, format="wav")

        # Calculate articulation score
        score1 = inference_wav(wav_filepath)
        # Calculate prosody score (assuming you have a separate model for this)
        score2 = inference_wav2(wav_filepath)  # You might need to modify this for prosody scoring
        
        total_score = score1 + score2

        return InferenceResult(
            total_score=total_score,
            articulation_score=score1,
            prosody_score=score2
        )

    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": f"Model file not found: {str(e)}"})

    except pydantic.ValidationError as e:
        return JSONResponse(status_code=422, content={"error": e.errors()}) 
    except Exception as e:  # Catch all other exceptions
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {str(e)}"})