import os
import json
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 사용자가 지정한 경로 및 상수 세팅
WAV_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\acc_voice_data"
JSON_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\acc_json_data"
IMAGE_SAVE_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project\Mel_image"
SAVE_DIR = r"C:\Users\KDT34\Documents\15-ai-data-project"

MAX_FILES = 1500      # 1차 테스트를 위한 최대 처리 파일 수 (전체 수행 시 숫자를 늘리거나 주석 처리)
SAMPLE_RATE = 16000   # 충격음과 사람 목소리를 모두 수집하는 나이퀴스트 표준 16kHz
N_MELS = 128          # Mel-Spectrogram 주파수 해상도 (세로축 이미지 크기)
FIXED_TIME = 3        # 핵심 상황(우당탕+대사)을 포착할 3초 규격화

# 필수 폴더 자동 생성
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

print("폴더 내에서 음성 파일(.wav)을 탐색 중입니다...")

# 2. 하위 폴더까지 포함하여 음성 파일 수집 (윈도우 대소문자 중복 제거)
wav_files = list(set(glob.glob(os.path.join(WAV_DIR, "**", "*.wav"), recursive=True)))
print(f"-> 중복 제거 후 발견된 순수 음성 파일 개수: {len(wav_files)}개")

X_list = []
y_list = []
count = 0

# Matplotlib 팝업 창이 뜨지 않도록 백엔드 설정 (고속 이미지 저장용)
plt.switch_backend('Agg')

# 3. 고속 Mel-Spectrogram 전처리 및 이미지 저장 루프 시작
for wav_path in tqdm(wav_files, desc="Mel-Spectrogram 변환 및 이미지 저장 중"):
    if count >= MAX_FILES:
        break
        
    # [A] 상대 경로 및 파일명 가공을 통해 매칭되는 JSON 파일 찾기
    relative_path = os.path.relpath(wav_path, WAV_DIR)
    base_name, _ = os.path.splitext(relative_path)
    
    # 음성 파일명 끝의 '_label' 제거 규칙 적용
    if base_name.endswith("_label"):
        base_name = base_name[:-6]
    elif base_name.endswith("_LABEL"):
        base_name = base_name[:-6]
        
    json_path = os.path.join(JSON_DIR, base_name + ".json")
    
    if not os.path.exists(json_path):
        continue  # JSON 파일이 없으면 패스
        
    try:
        # [B] JSON 파일에서 정답 정보(Label) 추출
        with open(json_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        category = str(meta_data.get("category", ""))
        label_text = str(meta_data.get("label", ""))
        
        # '낙상', '비명' 외에도 미끄러짐이나 떨어짐 대사 등이 감지되면 위험(1), 아니면 정상(0)
        if "낙상" in category or "낙상" in label_text or "비명" in label_text or "위급" in category:
            label = 1
        else:
            label = 0
            
        # [C] 오디오 로드 후 VAD 무음 제거 및 3초 고정 길이 규격화
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        # 앞뒤 무음 구간 자동으로 잘라내기 (우당탕 소리와 대사 시작점으로 정렬)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # 3초 분량의 샘플 수 계산
        required_samples = SAMPLE_RATE * FIXED_TIME
        if len(y_trimmed) < required_samples:
            y_final = np.pad(y_trimmed, (0, required_samples - len(y_trimmed)), 'constant')
        else:
            y_final = y_trimmed[:required_samples]
            
        # [D] Mel-Spectrogram 특징 추출 및 데시벨(dB) 변환
        mel_spec = librosa.feature.melspectrogram(y=y_final, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # [E] 시각화 이미지(.png) 저장 프로세스
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.axis('off') # AI 학습 및 깔끔한 저장을 위해 축 눈금 제거
        
        # 파일명을 그대로 본떠 Mel_image 폴더에 png로 저장
        img_name = f"{base_name}_label_{label}.png"
        plt.savefig(os.path.join(IMAGE_SAVE_DIR, img_name), bbox_inches='tight', pad_inches=0)
        plt.close() # 메모리 해제
        
        # 데이터 리스트에 추가
        X_list.append(mel_db)
        y_list.append(label)
        
        count += 1
        
    except Exception as e:
        continue

# 4. NumPy 배열 통합 및 파일 저장
if len(X_list) > 0:
    X_data = np.array(X_list)
    y_labels = np.array(y_list)

    print(f"\n" + "="*40)
    print(f"🎉 Mel-Spectrogram 변환 및 이미지 저장 완료!")
    print(f"최종 저장된 데이터 개수: {len(X_data)}개")
    print(f"데이터 행렬 형태(Shape): {X_data.shape} -> (데이터수, 128, 시간프레임)")
    print(f"위험 상황(1) 개수: {np.sum(y_labels)}개")
    print(f"정상 상황(0) 개수: {len(y_labels) - np.sum(y_labels)}개")
    print("="*40)

    np.save(os.path.join(SAVE_DIR, "X_data.npy"), X_data)
    np.save(os.path.join(SAVE_DIR, "y_labels.npy"), y_labels)
    print(f"AI 학습용 데이터셋이 {SAVE_DIR}에 성공적으로 통합 저장되었습니다.")
    print(f"시각화 이미지는 {IMAGE_SAVE_DIR} 폴더에서 확인할 수 있습니다.")
else:
    print("\n[알림] 조건에 매칭되는 음성(.wav)과 정답(.json) 파일 세트가 없습니다.")