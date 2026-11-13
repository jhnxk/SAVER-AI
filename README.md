# IEUM GitHub와 Render 배포 가이드

새 GitHub 저장소와 본인 Render 계정으로 전시용 링크를 만드는 방법입니다. 수정된 소스는 IEUM_Render_Deploy.zip에 들어 있습니다.

## 1. GitHub에 올리기

1. ZIP을 풀고 IEUM_Render_Deploy 폴더를 엽니다. 폴더 바로 아래에 Dockerfile, render.yaml, ai_server.py가 보여야 합니다.
2. GitHub에서 **New repository**를 선택합니다. 예시 이름은 **ieum-expo**입니다. **Private**으로 만들고 README, .gitignore, 라이선스 초기화 옵션은 비워 둡니다. 저장소가 Private이어도 Render가 만든 시연 주소는 브라우저에서 열 수 있습니다.
3. Mac 터미널에서 ZIP을 푼 폴더로 이동한 뒤 아래 명령을 실행합니다. YOUR_GITHUB_ID는 본인 아이디로 바꿉니다.

```bash
cd ~/Downloads/IEUM_Render_Deploy
git init -b main
git add .
git diff --cached --name-only
git commit -m "Prepare IEUM exhibition deployment"
git remote add origin https://github.com/YOUR_GITHUB_ID/ieum-expo.git
git push -u origin main
```

커밋 전 파일 목록에 실제 .env, Google 인증 JSON, dispatch_log.jsonl이 없어야 합니다. .env.example은 빈 값을 담은 예제입니다. 이번 ZIP에서 기존 전송 로그를 제외했습니다.

## 2. API 키 준비

| Render 변수 | 본인 계정에서 준비할 것 | 쓰는 곳 |
| --- | --- | --- |
| `SERVICE_KEY` | 공공데이터포털 **국립중앙의료원 전국 응급의료기관 정보 조회 서비스** 활용 신청 후 승인된 일반 인증키. 디코딩 키를 권장하며 인코딩 키도 처리합니다. | 가용병상과 장비의 주기 조회 |
| `GROQ_API_KEY` | Groq Console에서 발급 | 환자 발화 구조화 |
| `KAKAO_REST_API_KEY` | 카카오디벨로퍼스 앱의 REST API 키. 카카오모빌리티 길찾기 사용 설정과 쿼터 확인 | 도로 ETA와 상세 경로 |
| `KAKAO_JAVASCRIPT_KEY` | 카카오디벨로퍼스 앱의 JavaScript 키. 카카오맵 사용 설정 후 Render의 HTTPS 주소를 **JavaScript SDK 도메인**에 등록 | 브라우저 지도 |
| `GOOGLE_CLOUD_PROJECT` | 음성 시연을 켤 경우, Speech-to-Text API와 결제가 설정된 본인 Google Cloud 프로젝트 ID | 음성 전사 |
| `GOOGLE_APPLICATION_CREDENTIALS` | 음성 시연을 켤 경우, Render **Environment → Secret Files**에 google-stt.json을 만들고 Speech-to-Text 호출 권한이 있는 서비스 계정 JSON을 저장. 변수 값에는 `/etc/secrets/google-stt.json` 입력 | 서버 음성 인증 |

처음에는 공공데이터·Groq·카카오 키를 연결해 **텍스트 프리셋부터** 확인합니다. 음성까지 시연할 때 Google 설정을 추가하고 재배포합니다. HTTPS 접속 후 브라우저 마이크를 허용합니다. Google 스트리밍 기본 설정은 us와 chirp_3입니다. HIRA 자료는 함께 제공된 기준 파일을 읽으므로 전시 서비스 시작에 HIRA 키가 필요하지 않습니다.

## 3. Render에 올리기

1. [Render Dashboard](https://dashboard.render.com/)에서 **New → Blueprint**를 선택합니다.
2. 본인 GitHub를 연결하고 방금 만든 저장소의 접근을 허용합니다. 브랜치는 main, Blueprint 파일은 루트의 render.yaml입니다.
3. 2단계의 필수 키 네 개를 입력하고 비용과 설정을 확인한 다음 배포합니다. 파일은 **유료 Starter 웹 인스턴스 한 개**를 선택합니다. 서비스 이름 ieum-expo는 필요하면 본인 이름을 붙여 수정할 수 있습니다.
4. 서비스 상태가 **Live**가 되면 서비스 상단의 HTTPS 링크를 복사합니다. 그 주소를 카카오의 JavaScript SDK 도메인에 등록한 뒤 Render 링크를 다시 엽니다.
5. 소스를 수정한 뒤 같은 GitHub 브랜치에 push하면 Render의 자동 배포 설정에 따라 새 버전이 반영됩니다.

수동으로 만들려면 **New → Web Service**, 본인 저장소, **Docker**, **Starter**, 인스턴스 한 개를 선택합니다. **Health Check Path**는 /healthz입니다. Start Command는 Dockerfile이 제공하고 포트는 Render의 PORT를 읽습니다. render.yaml의 환경변수도 Render에 추가합니다.

무료로 화면만 먼저 볼 때는 수동 **Web Service → Free**에서 IEUM_LIVE_REFRESH=0으로 설정할 수 있습니다. 화면에 **과거 샘플 · 시연 전용**으로 표시됩니다. Render Free는 인바운드 트래픽이 15분 없으면 잠들고 다음 접속에 시작 시간이 생깁니다. 백그라운드 스레드만 돌아가는 동안은 인바운드 트래픽이 생기지 않습니다. 전시 때 연속 자동 갱신이 필요하면 Starter 구성을 사용합니다.

## 4. 배포 직후 확인하기

1. **/healthz**를 열어 HTTP 200과 `{"ok":true}`를 확인합니다. 이것은 웹 서버 준비 상태입니다.
2. **/api/runtime-db-status**를 열어 status, fresh_count, last_error를 봅니다. 정상 갱신은 **fresh**와 **0보다 큰 fresh_count**로 확인합니다. db_updated_at은 IEUM 조회 성공 시각이고 provider_latest_at은 유효한 기관 원본 시각 중 가장 최근 값입니다.
3. 메인에서 **뇌졸중** 프리셋을 선택하고 **AI 분석 · 병원 필터링**을 누릅니다. 환자 요구조건과 후보, 데이터 상태, 예상 이송시간 표시를 확인합니다. 모델 A → B → C → D를 바꾸면 같은 구조화 결과를 재사용합니다. 갱신된 DB는 다음 추천 요청부터 사용합니다.
4. 카카오 ETA가 붙은 행과 **약**으로 표시된 행을 구분합니다. 카카오 경로가 확인된 건수만 시간 기준 검증에 들어갑니다. 0/K는 확인 경로가 없다는 뜻입니다. 상세 지도는 선택 후 별도 시점에 조회하며 추천 시점의 표와 검증 수치는 함께 유지됩니다.
5. 전시 전 PC와 휴대폰에서 각각 확인합니다. 첫 추천, 같은 문장의 재추천, 모델 변경, 지도 경로, 음성을 켰다면 마이크까지 테스트합니다. DevTools Network의 /api/recommend-hospitals 응답에 있는 performance_metrics.stage_ms로 환자 분석·필터·경로·순위를 나누어 봅니다. 음성 시작부터 화면까지의 시간은 별도로 재고, 20회 이상 반복한 뒤 중앙값·P95와 실패 건수를 함께 기록합니다.

| 상태 또는 현상 | 확인할 것 |
| --- | --- |
| nemc_missing_service_key, nemc_result_* | 공공데이터 승인 서비스명, 키와 쿼터를 확인합니다. last_error에 반환 코드가 남습니다. |
| nemc_incomplete_pagination, nemc_inconsistent_total_count | 전국 조회의 XML totalCount와 페이지 응답을 확인합니다. 개발 환경에서도 Render와 같은 키로 점검합니다. |
| low_coverage_*_of_338 | 지역 조건을 생략한 전국 조회가 전체 범위를 반환하는지 확인합니다. 원본에 기관 ID가 신설·변경됐으면 주소 및 기관 정보를 대조한 뒤 교차표를 갱신합니다. 기준을 낮추기 전에 누락 원인을 확인합니다. |
| no_fresh_provider_observations, stale | 원본 hvidate가 현재 한국시각 기준 15분 이내인지 확인합니다. 서버가 방금 조회한 시각과 기관 원본 시각은 별개입니다. |
| degraded | 마지막 성공 자료 중 유효시간 이내인 행은 계속 쓰고 있습니다. 최근 실패가 이어지면 Render Logs와 API 응답을 확인합니다. |
| 지도는 뜨는데 ETA에 **약**이 붙음 | REST 키와 길찾기 쿼터, 후보 거리, 카카오 응답을 확인합니다. 다중 목적지 API 공식 반경 상한은 10km입니다. |
| 지도 자체가 로드되지 않음 | JavaScript 키와 HTTPS 도메인, 카카오맵 사용 설정을 확인합니다. |
| 음성이 비활성화됨 | Render Secret File과 GOOGLE_APPLICATION_CREDENTIALS를 확인합니다. 텍스트 프리셋은 계속 사용할 수 있습니다. |

기본 180초 주기는 한 페이지로 응답하면 약 **480회/일**입니다. 페이지가 늘면 호출 수도 늘어납니다. 서비스 승인 한도와 원본 갱신 주기를 먼저 확인한 뒤 조정합니다. 개별 기관의 15분 유효시간은 전시용 정책입니다. 진료과 자료는 기준 파일의 관측 시점에 해당하므로 별도로 갱신합니다.

## 5. 수정 파일과 이유

| 파일 | 반영한 수정 |
| --- | --- |
| `api_client.py` | 전국 가용병상 API를 지역 분할 없이 조회하고, 반환 코드와 모든 페이지를 검사합니다. 키 인코딩을 한 번만 적용하고 실패 메시지에서 인증 URL을 제외합니다. |
| `realtime_store.py` | 서버 시작 시 기준 파일을 한 번 읽습니다. 백그라운드에서 180초마다 갱신하고 검사를 통과한 스냅샷을 한 번에 교체합니다. 일부 응답 또는 장애 시 마지막 정상 스냅샷을 유지하며 기관별 원본 시각이 만료되면 해당 자원을 미확인 처리합니다. |
| `data/nemc_hpid_crosswalk.csv`, `build_hpid_crosswalk.py` | 첨부 DB 384곳 중 **정규화 병원명·시도가 일치하고 HPID 연결이 유일한 338곳**을 사용합니다. 기존 자료에서 삼성창원병원과 근로복지공단 창원병원에 같은 HPID가 배정된 사례가 있었습니다. 46곳의 실시간 자원은 미연결로 남겨 두었으며 주소·공식 기관 ID 대조가 필요합니다. |
| `ai_server.py` | 추천 요청은 메모리 스냅샷을 사용합니다. 최신 병상 관측이 없으면 추천 대기 응답을 반환합니다. 일부 조건 미확인 후보는 구분하고, 병상 자료만으로 중증환자 수용을 확정하지 않습니다. 카카오 확인 ETA와 근삿값을 구분하고 검증 지표에는 확인 ETA만 넣습니다. |
| `ai_server.py` | Model B는 후보 하나를 고르는 고정 목적함수이므로 반복 PuLP 풀이를 동일 목적함수의 내림차순 정렬로 교체했습니다. 동점이면 자원 점수와 기존 입력 순으로 정렬합니다. Groq 응답 형식과 예선 추론 강도 medium을 유지하고 같은 입력의 구조화 결과를 재사용합니다. low로 바꾸는 속도 실험은 항목별 정확도를 함께 재검증합니다. |
| `server.ai.html` | DB 상태, 미확인 후보, 추정 ETA, 확인 건수 분모를 표시합니다. 공개 시연은 가상 위치로 시작합니다. 예전 수용 신청 버튼은 **이송 경로 시뮬레이션**으로 표시하며 실제 병원 전송과 환자 로그 저장을 하지 않습니다. |
| `app.py` | 기존 Streamlit 관리는 로컬 유지보수용입니다. 로컬 동기화에는 양쪽에 같은 IEUM_SYNC_TOKEN이 필요합니다. Render의 자동 갱신 모드에서는 외부 DB 쓰기를 차단합니다. |
| `Dockerfile`, `render.yaml`, `gunicorn.conf.py`, `requirements-web.txt`, `.env.example` | 단일 Docker 웹 서비스와 한 Gunicorn 프로세스를 지정했습니다. 프로세스 내부의 HTTP·WebSocket 스레드는 같은 스냅샷을 사용합니다. |
| `tests/test_live_pipeline.py` | 외부 API 응답을 모의해 페이지 누락, 교차표 검증, 지연 중 이전 스냅샷, 재시작·만료, ETA 지표, 추천 흐름을 검증합니다. |

기본 Render 설정은 재시작 시 공공 API에서 다시 받아오므로 첫 갱신 동안 추천이 대기합니다. 재시작 후에도 직전 유효 스냅샷이 필요하면 유료 Render 서비스에 디스크를 붙이고 마운트 경로 아래에 IEUM_SNAPSHOT_DIR=/var/data/ieum을 지정합니다. 파일이 복원되어도 기관 원본 시각에 같은 유효시간을 적용합니다.

## 6. 검증 범위

- 로컬 자동 검증 7개가 통과했습니다. Gunicorn에서 웹·상태 API와 키 미설정 시 추천 대기 응답을 확인했습니다.
- 외부 응답을 대체하고 구조화된 가상 환자를 넘긴 Flask 테스트 20회에서 추천 API 중앙값 약 119ms, P95 약 182ms를 관측했습니다. STT, Groq, 카카오 네트워크, Render와 브라우저 시간은 측정에 포함되지 않았습니다.
- 예선 음성부터 추천 화면까지 평균 34.4초는 첨부 실험지의 기존 값입니다. 실제 API 키를 사용하는 전국 조회와 최신 배포본의 전체 소요시간은 배포 후 4단계에서 확인해야 합니다.
- 이 서비스는 전시용 연구 프로토타입입니다. 시험에는 프리셋이나 가상 사례를 사용합니다. 공개 데이터의 조건 충족은 병원 수용 확정과 별개이며, 실제 응급상황은 119에 연락합니다.

## 공식 문서

- [GitHub에 로컬 코드 올리기](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Render Blueprint](https://render.com/docs/blueprint-spec) · [무료 플랜](https://render.com/docs/free) · [환경변수와 Secret Files](https://render.com/docs/configure-environment-variables)
- [국립중앙의료원 전국 응급의료기관 정보 조회](https://www.data.go.kr/data/15000563/openapi.do) · [심평원 의료기관별 상세정보](https://www.data.go.kr/data/15001699/openapi.do)
- [카카오 앱 설정](https://developers.kakao.com/docs/latest/ko/app-setting/app) · [카카오모빌리티 다중 목적지 길찾기](https://developers.kakaomobility.com/guide/navi-api/destinations)
- [Groq 구조화 출력](https://console.groq.com/docs/structured-outputs) · [Google Speech-to-Text Chirp 3](https://cloud.google.com/speech-to-text/v2/docs/models/chirp-3)
