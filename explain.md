# Football Play Analysis System - 현황 요약

## 🎯 핵심 문제

Track ID Fragmentation: 경기장에 22명인데 50-100개의 track ID 생성됨

## 🛠 기술 스택

Detection/Tracking: YOLOv8n + ByteTrack
Team Classification: OpenAI CLIP (ViT-B/32) + K-means clustering
좌표 변환: OpenCV homography → Bird's Eye View
Pose Detection: YOLOv8n-pose (플레이 종료 감지)
OCR: Tesseract (등번호)

## 📊 현재 Tracking 설정

### bytetrack_extended.yaml

track_buffer: 900 frames (30초)
new_track_thresh: 0.5 (새 ID 생성 어렵게)
match_thresh: 0.5 (매칭 관대하게)

### detector_tracker.py

conf=0.25 (detection 딜레이)
iou=0.2 (tracking 안정성)
✅ 잘 작동하는 것
Player/Ball detection (YOLO)
BEV 변환으로 정확한 야드 계산
Ball carrier 식별 (유클리드 거리)
Play end detection (무릎 꿇음 감지)
CLIP 기반 팀 분류 (1-frame freeze)
Sideline 인원 필터링 (BEV boundary)
영상 출력 (result.mp4, bev.mp4, JSON)
❌ 현재 문제점

1. Track ID Fragmentation (치명적)
   원인: Detection 손실 → ByteTrack timeout → 같은 사람 재탐지 시 새 ID 생성
   영향:
   팀 분류 낭비 (100개 track에 CLIP 실행)
   같은 선수가 여러 팀 배정
   JSON 데이터 중복/불일치
2. 기타 제약
   Play type 항상 "RUN" (PASS 감지 제거됨)
   한 플레이만 처리 (배치 처리 없음)
   등번호 OCR 정확도 낮음 (측면 영상)
   포지션 추론 없음 (QB/RB/WR)
   Team freeze 후 수정 불가
   🔧 최근 시도 (롤백됨)
   CLIP Re-ID + Track ID Mapper 시도
   성능 문제로 제거 (한 명씩 순차 태깅)
   현재: ByteTrack만 사용
   💡 해결 방안 3가지
   Option A: Track ID Mapper (추천)

### 공간 근접도 + 팀 + CLIP embedding으로 ID 통합

- 가벼움 (기존 CLIP 활용)
- 실시간 처리 가능
- threshold 튜닝 필요
  Option B: ByteTrack 더 공격적으로
  track_buffer: 1800 (60초)
  new_track_thresh: 0.7 (더 높임)
  match_thresh: 0.3 (더 낮춤)
  Option C: Appearance Re-ID

### ResNet/CLIP으로 외형 매칭

- 정확도 높음
- 연산 오버헤드 큼
- 유사한 선수 혼동 위험
  📝 다음 작업 우선순위
  Immediate (ID Fragmentation 해결)
  Track ID Mapper 구현 (Option A)
  Track deduplication (플레이 종료 후)
  ByteTrack 더 튜닝 (Option B 병행)
  Medium (정확도 향상) 4. 포지션 추론 (formation pattern) 5. 멀티 플레이 배치 처리 6. 등번호 OCR 개선 (ensemble voting) Long-term (시스템 강화) 7. 실시간 preview UI 8. 설정 웹 인터페이스
  🎓 개발자에게 질문할 것
  Track ID Mapper vs ByteTrack 튜닝: 어느 쪽이 더 효과적?
  Appearance Re-ID 필요성: 연산 오버헤드 감수할 가치?
  CLIP embedding 재사용: 이미 있는데 Re-ID에 쓸 수 있나?
  ByteTrack 한계: track_buffer를 1800 (60초)까지 올려도 괜찮나?
  실시간성: ID Mapper 추가 시 FPS 영향도?
  📂 핵심 파일
  main.py (313줄): 메인 파이프라인
  detector_tracker.py (206줄): YOLO + ByteTrack
  play_analyzer.py (420줄): 상태 머신
  clip_classifier.py (444줄): 팀 분류
  bytetrack_extended.yaml: Tracking 설정
  📈 성능
  처리 속도: 10-30 FPS (GPU)
  CLIP 오버헤드: ~50ms/16 crops (30프레임마다)
  메모리: 적정 (500개 crop 이미지 저장)
  핵심 결론: 시스템 기반은 탄탄함. **Track ID Mapper (CLIP + 공간 근접도)**로 ID 통합이 최우선 과제.
