### 추론 돌리는 명령어 
cd server/
#score1 <- python3 inference_wav.py --wav score_input.m4a --lang en --label_type1 pron --label_type2 prosody --device cpu --dir_model ./model_ckpt/
#score2 <- python3 inference_wav.py --wav score_input.m4a --lang en --label_type1 pron --label_type2 articulation --device cpu --dir_model ./model_ckpt/
#최종 스코어 = score1+score2

### fast api 실행 코드
sh start.sh

### 설치 코드
pip install -r requirement.txt

### 필요한 파일
server - main.py
       - inference_wav.py
       - score_model.py
       - score_input.m4a
       - model_ckpt/
         - pron_articulation_checkpoint.pt
         - pron_prosody_checkpoint.pt