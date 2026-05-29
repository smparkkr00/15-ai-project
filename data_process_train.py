import os
import json
import glob
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# 1. 사용자 지정 경로 설정
JSON_DIR = r"C:\Users\KDT34\Documents\Data\1_fall_data\1_Training\1_train_json_data"
WAV_DIR = r"C:\Users\KDT34\Documents\Data\1_fall_data\1_Training\2_train_voice_data"
NPY_VOICE_DIR = r"C:\Users\KDT34\Documents\Data\1_fall_data\3_npy\1_Training\voice"
CSV_JSON_DIR = r"C:\Users\KDT34\Documents\Data\1_fall_data\3_npy\1_Training\json"

# 2. 전처리 스펙 설정 (16000Hz, 8초 패딩)
TARGET_SR = 16000
DURATION = 8  
TARGET_LENGTH = TARGET_SR * DURATION  # 128,000 샘플

# 저장 폴더 자동 생성
os.makedirs(NPY_VOICE_DIR, exist_ok=True)
os.makedirs(CSV_JSON_DIR, exist_ok=True)

# 정답지 정보를 모을 리스트
meta_records = []

# 3. JSON 파일 목록 긁어오기
json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다. 전처리를 시작합니다.")

# 4. 루프를 돌며 전처리 진행
for json_path in tqdm(json_files, desc="Processing Data"):
    try:
        # [4-1] JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # [4-2] 파일명 매칭 규칙 수정 (_label 추가)
        # JSON 파일명(예: 10.낙상_317257.json)에서 순수 이름만 추출
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # 실제 음성 파일명 정의: 이름 끝에 '_label.wav'를 붙임
        audio_name = f"{base_name}_label.wav"
        wav_path = os.path.join(WAV_DIR, audio_name)
        
        # 만약 실제 폴더에 해당 WAV 파일이 없다면 건너뛰기
        if not os.path.exists(wav_path):
            # 혹시나 해서 JSON 내부 안의 오디오 ID 정보로도 2차 확인해보는 방어 코드
            audio_id = meta['annotations'][0].get('audio_id', '')
            if audio_id:
                audio_name = f"{audio_id}.wav"
                wav_path = os.path.join(WAV_DIR, audio_name)
            
            # 둘 다 없으면 진짜 파일이 없는 것이므로 스킵
            if not os.path.exists(wav_path):
                continue
            
        # 가위질할 타겟 구간 (area의 start, end)
        annotation = meta['annotations'][0]
        start_time = annotation['area']['start']
        end_time = annotation['area']['end']
        
        # 세부 정보 추출
        sub_category = annotation.get('subCategory', annotation['categories'].get('category_03', ''))
        note = annotation.get('note', '')

        # [4-3] 오디오 전처리 (Librosa 활용)
        # 리샘플링(16kHz) 및 모노 변환하면서 로드
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        
        # JSON에 적힌 구간(start ~ end)만 자르기 (Crop)
        start_sample = int(start_time * TARGET_SR)
        end_sample = int(end_time * TARGET_SR)
        y_cropped = y[start_sample:end_sample]
        
        # 볼륨 정규화 (Peak Normalization)
        if len(y_cropped) > 0:
            max_val = np.max(np.abs(y_cropped))
            if max_val > 0:
                y_cropped = y_cropped / max_val
        
        # 8초 길이 맞추기 (Padding / Truncation)
        if len(y_cropped) < TARGET_LENGTH:
            y_final = np.pad(y_cropped, (0, TARGET_LENGTH - len(y_cropped)), mode='constant')
        else:
            y_final = y_cropped[:TARGET_LENGTH]

        # 멜-스펙트로그램(Mel-Spectrogram) 특징 추출
        mel_spec = librosa.feature.melspectrogram(
            y=y_final, sr=TARGET_SR, n_fft=1024, hop_length=512, n_mels=128
        )
        # 로그 스케일(dB) 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # [4-4] npy 파일 저장 (저장할 때는 직관적으로 base_name.npy로 저장)
        npy_filename = f"{base_name}.npy"
        npy_save_path = os.path.join(NPY_VOICE_DIR, npy_filename)
        np.save(npy_save_path, mel_spec_db)

        # [4-5] 정답지 리스트에 기록 추가 (사고발생이므로 일괄 라벨 '1')
        meta_records.append({
            "npy_filename": npy_filename,
            "label": 1,
            "label_name": "사고발생(낙상)",
            "detail": sub_category,
            "note": note
        })

    except Exception as e:
        print(f"\n파일 처리 중 에러 발생 ({os.path.basename(json_path)}): {e}")
        continue

# 5. 모든 파일 처리가 끝나면 통합 정답지를 CSV 파일로 저장
if meta_records:
    df = pd.DataFrame(meta_records)
    csv_save_path = os.path.join(CSV_JSON_DIR, "train_labels.csv")
    df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"\n✨ 전처리 완료! 총 {len(meta_records)}개의 npy 파일 생성 완료.")
    print(f"📝 통합 정답지 저장 완료: {csv_save_path}")
else:
    print("\n❌ 전처리된 데이터가 없습니다. 경로 및 파일명을 다시 확인해 주세요.")