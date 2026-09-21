# SAVER + 지도팀 카카오맵 통합 v7

이번 버전은 지도팀이 준 `카카오맵-전국병원(09.12)`의 네비게이션 화면/시뮬레이션 로직을 SAVER 안에서 직접 호출하도록 합친 버전입니다.

## 반영 범위
- 기존 SAVER 추천/DB/환자 위치 랜덤 생성/저장 후 SAVER 이동 흐름 유지
- `병원 사전 수용 신청 및 이송 시작` 클릭 시 지도팀 `trafficSimulation.js`의 `startNavigation(hospital)` 호출
- 지도팀 네비게이션 화면 HTML 구조와 CSS를 SAVER에 추가
- 선택한 병원 하나의 경로만 전체 화면 네비게이션 모드로 표시

## 실행
```bash
python3 ai_server.py
```

브라우저:
```text
http://localhost:5050
```

`.env`에는 아래 키가 필요합니다.
```env
KAKAO_REST_API_KEY=...
KAKAO_JAVASCRIPT_KEY=...
PORT=5050
```
