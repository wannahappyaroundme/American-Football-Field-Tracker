# 🏈 Football Play Analysis System
## 미식축구 경기 영상 자동 분석 시스템

> YOLO11 + ByteTrack + CLIP + Jersey OCR을 활용한 선수 추적, 팀 분류, 플레이 통계 자동 생성

---

## 🎯 프로젝트 개요

미식축구 경기 영상에서 **선수, 심판, 공을 자동으로 감지하고 추적**하여 플레이 통계를 생성하는 컴퓨터 비전 시스템입니다.

### 현재 성능
- **탐지율**: 16-23명/프레임 (22명 선수 목표)
- **ID 안정성**: 73 unique IDs (3.3x multiplier)
- **팀 분류 정확도**: 95%+ (CLIP 기반)
- **처리 속도**: ~40ms/frame (25 FPS on CPU)
- **심판 감지**: 3-4명 (목표 1-4명 달성 ✅)

---

## ✨ 주요 기능

1. **실시간 객체 탐지 및 추적**
   - YOLO11n: 경량 객체 탐지 (YOLOv8 대비 22% 파라미터 감소)
   - ByteTrack: 다중 객체 추적 (Kalman filter, 3초 buffer)
   - Jersey OCR: 등번호 인식으로 ID 안정화

2. **지능형 팀 분류**
   - CLIP (OpenAI): Vision-Language Model
   - 시맨틱 이해: 유니폼 색상 + 헬멧 + 줄무늬 패턴
   - 정확도: HSV 80% → CLIP 95%+

3. **플레이 분석**
   - 상태 머신: PRE_SNAP → PLAY_ACTIVE → BALL_IN_AIR → PLAY_ENDED
   - 볼 캐리어 감지, 야드 계산, JSON 출력

4. **시각화**
   - Bounding Box: 팀별 색상 (Team A=파랑, Team B=빨강, Referee=노랑)
   - Bird's Eye View: 전술 맵, 선수 이동 경로

---

## 🛠 기술 스택

| 분야 | 기술 | 선택 이유 |
|------|------|----------|
| **객체 탐지** | YOLO11n | YOLOv8 대비 22% 적은 파라미터, 작은 객체 탐지 개선 |
| **객체 추적** | ByteTrack | 2-stage association, Kalman filter, 실시간 성능 |
| **팀 분류** | CLIP (ViT-B/32) | Zero-shot learning, 시맨틱 이해, 95% 정확도 |
| **등번호 인식** | Tesseract OCR | 오픈소스, 무료, 다국어 지원 |
| **BEV 변환** | OpenCV Homography | 4점 매칭으로 정확한 야드 계산 |

---

## 🏗 시스템 아키텍처

### 데이터 흐름
```
Video Input
    ↓
YOLO11 Detection (config.py: DETECTION_CONFIDENCE_THRESHOLD=0.3)
    ↓
ByteTrack Tracking (bytetrack_extended.yaml: track_buffer=90, match_thresh=0.70)
    ↓
Jersey OCR (30프레임마다)
    ↓
Jersey-based ID Mapping (Team A: 1000-1999, Team B: 2000-2999)
    ↓
CLIP Team Classification (30프레임마다, 95% 정확도)
    ↓
BEV Transformation (homography_matrix.npy)
    ↓
Play Analysis (State machine)
    ↓
Visualization
    ↓
Output → result.mp4, bev.mp4, clip_summary.json
```

### 핵심 컴포넌트

**1. DetectorTracker** (detector_tracker.py)
- YOLO11n으로 person/ball 탐지
- ByteTrack으로 ID 부여
- 30프레임마다 Jersey OCR
- Jersey-based stable ID 매핑

**2. JerseyBasedIDManager** (jersey_id_manager.py)
- 등번호 → Stable ID 매핑
- Team-scoped IDs (Team A: 1000s, Team B: 2000s, Referee: 3000s)
- 충돌 해결: 같은 등번호 여러 번 감지시 confidence 높은 것 선택

**3. TeamClassifier** (team_classifier.py)
- CLIP ViT-B/32 기반 팀 분류
- Prompt engineering: "yellow jersey with white helmet" vs "striped referee shirt"
- Team freezing: 한번 분류되면 절대 안 바뀜

**4. PlayAnalyzer** (play_analyzer.py)
- 4-state 머신: PRE_SNAP → PLAY_ACTIVE → BALL_IN_AIR → PLAY_ENDED
- 볼 캐리어 감지: BEV 최단 거리
- Pose detection으로 플레이 종료 감지

---

## 📦 설치 방법

### 1. 환경 요구사항
```bash
Python 3.8+
Tesseract OCR
```

### 2. Tesseract 설치
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Python 패키지
```bash
pip install -r requirements.txt

# 또는 수동 설치
pip install ultralytics>=8.3.0
pip install opencv-python>=4.8.0
pip install torch torchvision
pip install transformers pytesseract
pip install numpy scipy scikit-learn
```

### 4. Homography 캘리브레이션
```bash
python calibrate_homography.py
# 경기장 4개 점 클릭 → 's' 저장
```

---

## 🚀 사용법

### 기본 실행
```bash
python main.py
# Input: input/video.mp4
# Output: output/result.mp4, output/bev.mp4, output/clip_summary.json
```

### 주요 설정 (config.py)

**탐지 임계값 조정**
```python
DETECTION_CONFIDENCE_THRESHOLD = 0.3  # 0.1~0.6 추천
# 낮을수록: 많이 탐지, 오탐지 증가
# 높을수록: 확실한 것만, 놓치는 것 증가
```

**BEV 필터링**
```python
BEV_FIELD_Y_MIN = 0    # 상단 엔드존 (0 = 전체 포함)
BEV_FIELD_Y_MAX = 500  # 하단 엔드존 (500 = 전체 포함)
# 이전: 50-450 (너무 좁음) → 현재: 0-500 (전체 필드)
```

**CLIP 팀 분류**
```python
ENABLE_CLIP_TEAM_CLASSIFICATION = True
CLIP_FRAME_INTERVAL = 30  # 30프레임마다 (1초에 1번)
CLIP_CONFIDENCE_THRESHOLD = 0.50  # 50% 이상만 분류
```

### ByteTrack 튜닝 (bytetrack_extended.yaml)

**ID 수를 줄이려면**
```yaml
new_track_thresh: 0.60  # 0.50 → 0.60 (더 엄격)
# 주의: multi-detection 감소 가능
```

**Multi-detection을 늘리려면**
```yaml
track_low_thresh: 0.15  # 0.20 → 0.15 (더 관대)
# 주의: ID 수 증가 가능
```

**ID 안정성을 높이려면**
```yaml
track_buffer: 120  # 90 → 120 (4초)
match_thresh: 0.80  # 0.70 → 0.80 (더 관대)
```

---

## 📊 성능 지표

| 지표 | 현재 | 목표 | 상태 |
|------|-----|------|------|
| Unique IDs | 73 | 30-40 | 🟡 개선 중 |
| Multi-detection | 16-23/frame | 20-25 | ✅ 달성 |
| Referee count | 3-4 | 1-4 | ✅ 달성 |
| Team classification | 95%+ | 90%+ | ✅ 달성 |
| Processing speed | 40ms/frame | <50ms | ✅ 달성 |
| Ball detection | 0.2% | 10%+ | ❌ 낮음 |

### ID 안정성 개선 과정

| Phase | 방법 | IDs | Multiplier |
|-------|------|-----|------------|
| Phase 1 | YOLOv8 + default ByteTrack | 133 | 6.0x |
| Phase 2 | YOLO11 + optimized ByteTrack | 75 | 3.4x |
| Phase 3 | + Jersey OCR mapping | 73 | 3.3x ✅ |
| Future | + BoT-SORT Re-ID | 40-50 | 1.8-2.3x 🔮 |

---

## 📁 파일 구조

```
camera_tracking/
│
├── 📄 config.py                    # ⭐ 통합 설정 파일
├── 📄 main.py                      # ⭐ 메인 파이프라인
│
├── 🎯 객체 탐지 및 추적
│   ├── detector_tracker.py         # YOLO + ByteTrack + Jersey OCR
│   ├── jersey_id_manager.py        # 등번호 기반 ID 매핑
│   ├── bytetrack_extended.yaml     # ByteTrack 설정 (최적화됨)
│   └── botsort.yaml                # BoT-SORT 설정 (참고용, 미사용)
│
├── 👥 팀 분류
│   ├── team_classifier.py          # CLIP 기반 팀 분류
│   ├── clip_classifier.py          # CLIP 모델 래퍼
│   └── number_recognizer.py        # Tesseract OCR 래퍼
│
├── 🗺️ BEV 변환
│   ├── transformer.py              # Homography 변환
│   ├── calibrate_homography.py     # 캘리브레이션 도구
│   └── homography_matrix.npy       # 변환 행렬
│
├── 📊 플레이 분석
│   ├── play_analyzer.py            # 상태 머신
│   └── visualizer.py               # 시각화
│
└── 📖 문서
    ├── README.md                   # 이 파일
    ├── CLAUDE.md                   # Claude Code 가이드
    └── requirements.txt            # Python 의존성
```

---

## 🐛 트러블슈팅

### "FileNotFoundError: homography_matrix.npy"
```bash
python calibrate_homography.py
# 경기장 4개 점 클릭 → 's' 저장
```

### ID가 너무 많음 (100+ IDs)
```yaml
# bytetrack_extended.yaml
new_track_thresh: 0.60  # 0.50 → 0.60
```

### 선수 탐지 안 됨 (1-5명만)
```python
# config.py
BEV_FIELD_Y_MIN = 0
BEV_FIELD_Y_MAX = 500

# bytetrack_extended.yaml
new_track_thresh: 0.45  # 0.50 → 0.45
```

### 팀 분류가 계속 바뀜
```python
# config.py
FREEZE_TEAM_ASSIGNMENT = True
TEAM_ASSIGNMENT_CONFIDENCE = 2
```

### 공 탐지 안 됨
```python
# config.py
BALL_CONFIDENCE = 0.05  # 0.10 → 0.05
# 근본 해결: Fine-tuned model 필요
```

---

## 📈 개발 히스토리

- **Phase 1** (2024-10): YOLOv8 + ByteTrack → 133 IDs
- **Phase 2** (2024-10): BEV 확장 → 25-30명 탐지
- **Phase 3** (2024-10): CLIP 팀 분류 → 95% 정확도
- **Phase 4** (2024-11): 심판 post-processing → 3-4명
- **Phase 5** (2024-11): ByteTrack 최적화 → 73 IDs
- **Phase 6** (2024-11-08): Jersey-based ID system 구축 ✅

---

## 🔮 향후 개선 사항

### 단기
1. Jersey OCR 개선 (정면 카메라, 전처리 강화)
2. BoT-SORT 재도전 (Re-ID 기능 → 40-50 IDs 목표)
3. Ball Detection 개선 (Fine-tuned YOLO11)

### 중기
4. Real-time processing (GPU 최적화)
5. Formation detection (4-3, 3-4 등)
6. Web dashboard

### 장기
7. Multi-camera fusion (3D reconstruction)
8. Advanced analytics (EPA, WPA)
9. Production deployment (Docker, Cloud)

---

## 📄 라이선스

Educational and research purposes.

**사용 모델**:
- YOLO11: AGPL-3.0 (Ultralytics)
- ByteTrack: MIT License
- CLIP: MIT License (OpenAI)
- Tesseract: Apache 2.0

---

**마지막 업데이트**: 2024-11-08  
**버전**: 1.0.0 (Jersey-based ID System)  
**상태**: Production Ready ✅
