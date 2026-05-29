import os
import json
import glob
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# 1. 상위 기본 경로 설정
NOISE_BASE_DIR = r"C:\Users\KDT34\Documents\Data\2_noise_data\1_Training\1_train_noise_data"
JSON_BASE_DIR = r"C:\Users\KDT34\Documents\Data\2_noise_data\1_Training\2_train_json_data"
NPY_SAVE_DIR = r"C:\Users\KDT34\Documents\Data\2_noise_data\3_npy\1_Traing\noise"
CSV_SAVE_DIR = r"C:\Users\KDT34\Documents\Data\2_noise_data\3_npy\1_Traing\json"

# 2. 전처리 스펙 설정 (낙상 데이터셋과 완벽히 일치)
TARGET_SR = 16000
DURATION = 8  
TARGET_LENGTH = TARGET_SR * DURATION  # 128,000 샘플
LIMIT_PER_FOLDER = 1500  # 폴더당 최대 처리 개수

# 저장 폴더 생성
os.makedirs(NPY_SAVE_DIR, exist_ok=True)
os.makedirs(CSV_SAVE_DIR, exist_ok=True)

# 10개 세부 하위 경로 리스트
sub_folders = [
    r"1.자동차\1.차량경적", r"1.자동차\2.차량사이렌", r"1.자동차\3.차량주행음",
    r"2.이륜자동차\4.이륜차경적", r"2.이륜자동차\5.이륜차주행음",
    r"4.열차\9.지하철",
    r"5.충격\10.발소리", r"5.충격\11.가구소리",
    r"6.가전\12.청소기", r"6.가전\13.세탁기"
]

meta_records = []

# 3. 10개 폴더 루프 돌기
for sub_path in sub_folders:
    current_json_dir = os.path.join(JSON_BASE_DIR, sub_path)
    current_noise_dir = os.path.join(NOISE_BASE_DIR, sub_path)
    
    # 해당 폴더의 JSON 파일들 검색 및 1500개 제한
    json_files = glob.glob(os.path.join(current_json_dir, "*.json"))
    json_files = json_files[:LIMIT_PER_FOLDER]
    
    folder_name = os.path.basename(sub_path)
    print(f"\n📂 [{folder_name}] 폴더에서 {len(json_files)}개의 파일 처리를 시작합니다.")
    
    # 각 폴더 내부 파일 전처리 루프
    for json_path in tqdm(json_files, desc=f"Processing {folder_name}"):
        try:
            # [3-1] JSON 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # [3-2] 끝에 _1이 붙는 WAV 파일명 규칙 반영
            annotation = meta['annotations'][0]
            
            # 1순위 규칙: annotations 내부의 labelName 필드 활용 (가장 정확)
            audio_name = annotation.get('labelName', '')
            
            # 2순위 방어 코드: 만약 labelName이 없다면 JSON 파일명 끝에 _1을 붙여서 유추
            if not audio_name:
                base_json_name = os.path.splitext(os.path.basename(json_path))[0]
                audio_name = f"{base_json_name}_1.wav"
                
            if not audio_name.endswith('.wav'):
                audio_name += '.wav'
                
            wav_path = os.path.join(current_noise_dir, audio_name)
            
            # 3순위 방어 코드: 혹시나 _1이 안 붙은 원본 파일명이 존재할 수도 있으니 2차 확인
            if not os.path.exists(wav_path):
                alt_name = meta['audio']['fileName']
                if not alt_name.endswith('.wav'): alt_name += '.wav'
                alt_wav_path = os.path.join(current_noise_dir, alt_name)
                if os.path.exists(alt_wav_path):
                    wav_path = alt_wav_path
                else:
                    # 모든 경로에 파일이 진짜 없으면 스킵
                    continue
                
            # 가위질 구간 및 메타데이터 추출
            start_time = annotation['area']['start']
            end_time = annotation['area']['end']
            sub_cat = annotation.get('subCategory', '')
            cat_3 = annotation['categories'].get('category_03', '')

            # [3-3] 오디오 시그널 전처리
            y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            
            # 시간 크롭 (Start ~ End)
            start_sample = int(start_time * TARGET_SR)
            end_sample = int(end_time * TARGET_SR)
            y_cropped = y[start_sample:end_sample]
            
            # 볼륨 정규화 (Peak Normalization)
            if len(y_cropped) > 0:
                max_val = np.max(np.abs(y_cropped))
                if max_val > 0:
                    y_cropped = y_cropped / max_val
            
            # 8초 맞추기 (Padding / Truncation)
            if len(y_cropped) < TARGET_LENGTH:
                y_final = np.pad(y_cropped, (0, TARGET_LENGTH - len(y_cropped)), mode='constant')
            else:
                y_final = y_cropped[:TARGET_LENGTH]

            # 로그 멜-스펙트로그램 추출
            mel_spec = librosa.feature.melspectrogram(
                y=y_final, sr=TARGET_SR, n_fft=1024, hop_length=512, n_mels=128
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # [3-4] npy 저장 (중복 방지를 위해 폴더명과 맵핑된 고유 이름 생성)
            # 나중에 매칭하기 쉽도록 확장자를 제외한 순수 오디오 파일명을 npy 이름으로 활용합니다.
            pure_audio_name = os.path.splitext(os.path.basename(wav_path))[0]
            npy_filename = f"{folder_name}_{pure_audio_name}.npy"
            npy_save_path = os.path.join(NPY_SAVE_DIR, npy_filename)
            np.save(npy_save_path, mel_spec_db)

            # [3-5] 일상소음 정답지에 추가 (라벨 '0')
            meta_records.append({
                "npy_filename": npy_filename,
                "label": 0,
                "label_name": "일상소음",
                "detail": sub_cat if sub_cat else cat_3,
                "folder_category": folder_name
            })

        except Exception as e:
            print(f"\n오류 발생 - 파일: {os.path.basename(json_path)} | 사유: {e}")
            continue

# 4. 일상소음 통합 정답지 생성 (noise_train_labels.csv)
if meta_records:
    df = pd.DataFrame(meta_records)
    csv_save_path = os.path.join(CSV_SAVE_DIR, "noise_train_labels.csv")
    df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"\n✨ 일상소음 전처리 대성공! 총 {len(meta_records)}개의 npy 파일 빌드 완료.")
    print(f"📝 통합 일상소음 정답지 저장 완료: {csv_save_path}")
else:
    print("\n❌ 파일 전처리에 실패했습니다. 파일명 규칙이나 경로 구조를 다시 확인해 주세요.")