import os
import json
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 경로 및 상수 세팅
WAV_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\acc_voice_data"
JSON_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\acc_json_data"
IMAGE_SAVE_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\MFCC_image"
SAVE_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project"

MAX_FILES = 1500      # 1차 검증용 테스트 개수
SAMPLE_RATE = 16000   # 나이퀴스트 표준 16kHz
N_MELS = 128          # 1단계: Mel-Spectrogram 주파수 해상도
N_MFCC = 20           # 2단계: 최종 압축하여 남길 MFCC 계수 개수
FIXED_TIME = 3        # 3초 규격화

# 필수 폴더 자동 생성
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

print("폴더 내에서 음성 파일(.wav)을 탐색 중입니다...")
wav_files = list(set(glob.glob(os.path.join(WAV_DIR, "**", "*.wav"), recursive=True)))
print(f"-> 중복 제거 후 발견된 순수 음성 파일 개수: {len(wav_files)}개")

X_list = []
y_list = []
count = 0

# 고속 이미지 저장을 위해 백엔드 설정
plt.switch_backend('Agg')

# 2. 전처리 루프 시작
for wav_path in tqdm(wav_files, desc="Mel 기반 MFCC 변환 및 이미지 저장 중"):
    if count >= MAX_FILES:
        break
        
    relative_path = os.path.relpath(wav_path, WAV_DIR)
    base_name, _ = os.path.splitext(relative_path)
    
    if base_name.endswith("_label") or base_name.endswith("_LABEL"):
        base_name = base_name[:-6]
        
    json_path = os.path.join(JSON_DIR, base_name + ".json")
    if not os.path.exists(json_path):
        continue
        
    try:
        # [A] JSON 정답 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        category = str(meta_data.get("category", ""))
        label_text = str(meta_data.get("label", ""))
        
        if "낙상" in category or "낙상" in label_text or "비명" in label_text or "위급" in category:
            label = 1
        else:
            label = 0
            
        # [B] 오디오 로드 및 VAD 무음 제거 후 3초 맞춤
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        required_samples = SAMPLE_RATE * FIXED_TIME
        if len(y_trimmed) < required_samples:
            y_final = np.pad(y_trimmed, (0, required_samples - len(y_trimmed)), 'constant')
        else:
            y_final = y_trimmed[:required_samples]
            
        # [C] 🌟 Mel-Spectrogram 추출 후 MFCC로 최종 변환 수행
        # 파워 스펙트럼 계산 -> Mel 필터뱅크 적용 -> 로그 변환 -> DCT 연산 파이프라인입니다.
        mel_spec = librosa.feature.melspectrogram(y=y_final, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Mel-Spectrogram 에너지 배열로부터 인간 청각 특성이 극대화된 20개 계수 추출
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
        
        # [D] MFCC 주파수 시각화 이미지(.png) 저장
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', cmap='coolwarm')
        plt.axis('off') # 경계선 및 축 제거
        
        img_name = f"{base_name}_mfcc_{label}.png"
        plt.savefig(os.path.join(IMAGE_SAVE_DIR, img_name), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        X_list.append(mfcc)
        y_list.append(label)
        count += 1
        
    except Exception as e:
        continue

# 3. 데이터셋 통합 저장
if len(X_list) > 0:
    X_data = np.array(X_list)
    y_labels = np.array(y_list)

    print(f"\n" + "="*40)
    print(f"🎉 MFCC 변환 및 이미지 저장 완료!")
    print(f"최종 저장된 데이터 개수: {len(X_data)}개")
    print(f"데이터 행렬 형태(Shape): {X_data.shape} -> (데이터수, 20, 시간프레임)")
    print(f"위험 상황(1) 개수: {np.sum(y_labels)}개")
    print(f"정상 상황(0) 개수: {len(y_labels) - np.sum(y_labels)}개")
    print("="*40)

    # 요청하신 이름 규칙대로 npy 파일 저장
    np.save(os.path.join(SAVE_DIR, "X_MFCC.npy"), X_data)
    np.save(os.path.join(SAVE_DIR, "y_MFCC.npy"), y_labels)
    print(f"새로운 MFCC 데이터셋 파일이 {SAVE_DIR}에 정상 저장되었습니다.")