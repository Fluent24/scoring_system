import torchaudio
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.ASR import EncoderDecoderASR
import tempfile
import subprocess
from pydub import AudioSegment
import tempfile
import os
import sys
sys.path.append('/usr/bin/ffmpeg')

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

@app.post("/infer/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 일시적으로 파일 저장
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_file:
        temp_file.write(await file.read())
        m4a_filepath = temp_file.name

    # M4A 파일을 WAV 형식으로 변환
    wav_filepath = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    track = AudioSegment.from_file(m4a_filepath, format='m4a')
    track.export(wav_filepath, format='wav')

    try:
        # 추론 코드 실행 (첫 번째 매개변수 세트)
        result1 = subprocess.run(
            ['python3', 'inference_wav.py', '--wav', wav_filepath, '--lang', 'en', '--label_type1', 'pron', '--label_type2', 'articulation', '--device', 'cpu', '--dir_model', '../model_ckpt'],
            capture_output=True,
            text=True,
            check=True
        )
        score1 = float(result1.stdout.strip())

        # 추론 코드 실행 (두 번째 매개변수 세트)
        result2 = subprocess.run(
            ['python3', 'inference_wav.py', '--wav', wav_filepath, '--lang', 'en', '--label_type1', 'pron', '--label_type2', 'prosody', '--device', 'cpu', '--dir_model', '../model_ckpt'],
            capture_output=True,
            text=True,
            check=True
        )
        score2 = float(result2.stdout.strip())

        # 두 점수 합산
        total_score = score1 + score2

        return {"total_score": total_score, "articulation_score": score1, "prosody_score": score2}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}
    finally:
        # 임시 파일 삭제
        try:
            os.remove(m4a_filepath)
            os.remove(wav_filepath)
        except Exception as e:
            return {"error": f"Error cleaning up temporary files: {str(e)}"}
