// =====================================================
// trafficSimulation.js
// 카카오맵 실시간 교통 내비게이션 시뮬레이션
//
// 주요 기능
// 1. 카카오 실제 자동차 경로 사용
// 2. 카카오 traffic_speed / traffic_state 반영
// 3. 실제 도로 좌표를 따라 응급차량 시뮬레이션
// 4. X1 ~ X10 시뮬레이션 배속 조절
// 5. 60초마다 현재 위치에서 교통정보 재조회
//
// 변경 사항
// - 가상 신호등 기능 완전 삭제
// - 신호로 인한 감속/정차 삭제
// - 교통 정체/서행/지체는 계속 반영
// =====================================================

(function () {

    "use strict";


    // =====================================================
    // 기본 설정
    // =====================================================



    // 시뮬레이션 계산 주기
    const TICK_INTERVAL_MS = 250;


    // 카카오 교통 경로를 실제 시간 기준 60초마다 다시 조회
    const TRAFFIC_REFRESH_INTERVAL_MS =
        60 * 1000;


    // ⭐ 시뮬레이션 배속
    // 기본값 X1
    let SIMULATION_MULTIPLIER = 1;


    // =====================================================
    // 상태
    // =====================================================

    const state = {

        active: false,

        hospital: null,

        route: null,

        model: null,

        map: null,

        trafficLines: [],

        hospitalMarker: null,

        vehicleOverlay: null,

        vehicleElement: null,

        timerId: null,

        lastTickAt: null,

        simulationClockSeconds: 0,

        traveledMeters: 0,

        currentSpeedKmh: 0,

        currentPosition: null,

        currentHeading: 0,

        refreshing: false,

        lastRefreshAt: 0,

// 지도 자동 이동 제어
lastMapFollowAt: 0,
lastMapFollowPosition: null,

        // 현재 표시 중인 다음 안내를 기억
        currentGuideKey: null,

        // 다음 안내 팝업 타이머
        guidePopupTimer: null,

        reroutePopupTimer: null


    };


    // =====================================================
    // 공통 함수
    // =====================================================

    function setText(
        id,
        value
    ) {

        const element =
            document.getElementById(id);


        if (!element) {
            return;
        }


        element.textContent =
            value ?? "";

    }


    // =====================================================
    // 신호 관련 UI 완전 제거
    // =====================================================

    function removeTrafficSignalUI() {

        // 기존 신호 상태 표시 영역
        const signalState =
            document.getElementById(
                "traffic-signal-state"
            );


        if (signalState) {

            const parent =
                signalState.parentElement;


            if (parent) {

                parent.style.display =
                    "none";

            }
            else {

                signalState.style.display =
                    "none";

            }

        }


        // 기존 신호 마커가 혹시 남아있다면 제거
        const signalMarkers =
            document.querySelectorAll(
                ".simulation-signal-marker"
            );


        signalMarkers.forEach(
            marker => {

                marker.remove();

            }
        );

    }


    // =====================================================
    // 라디안 변환
    // =====================================================

    function toRadians(
        degrees
    ) {

        return degrees *
            Math.PI /
            180;

    }


    // =====================================================
    // 거리 계산
    // =====================================================

    function distanceMeters(
        lat1,
        lng1,
        lat2,
        lng2
    ) {

        const R =
            6371000;


        const dLat =
            toRadians(
                lat2 - lat1
            );


        const dLng =
            toRadians(
                lng2 - lng1
            );


        const a =
            Math.sin(dLat / 2) *
            Math.sin(dLat / 2) +

            Math.cos(
                toRadians(lat1)
            ) *

            Math.cos(
                toRadians(lat2)
            ) *

            Math.sin(dLng / 2) *
            Math.sin(dLng / 2);


        const c =
            2 *
            Math.atan2(
                Math.sqrt(a),
                Math.sqrt(1 - a)
            );


        return R * c;

    }


    // =====================================================
    // 방향 계산
    // =====================================================

    function bearingDegrees(
        lat1,
        lng1,
        lat2,
        lng2
    ) {

        const y =
            Math.sin(
                toRadians(
                    lng2 - lng1
                )
            ) *
            Math.cos(
                toRadians(lat2)
            );


        const x =
            Math.cos(
                toRadians(lat1)
            ) *
            Math.sin(
                toRadians(lat2)
            ) -

            Math.sin(
                toRadians(lat1)
            ) *
            Math.cos(
                toRadians(lat2)
            ) *
            Math.cos(
                toRadians(
                    lng2 - lng1
                )
            );


        const angle =
            Math.atan2(
                y,
                x
            ) *
            180 /
            Math.PI;


        return (
            angle + 360
        ) % 360;

    }


    // =====================================================
    // 거리 표시
    // =====================================================

    function formatDistance(
        meters
    ) {

        if (
            !Number.isFinite(
                meters
            )
        ) {

            return "-";

        }


        if (
            meters >= 1000
        ) {

            return (
                meters / 1000
            ).toFixed(1) +
            " km";

        }


        if (
            meters < 50
        ) {

            const rounded =
                Math.max(
                    0,
                    Math.round(
                        meters / 5
                    ) * 5
                );


            return rounded + " m";

        }


        const rounded =
            Math.round(
                meters / 10
            ) * 10;


        return rounded + " m";

    }


    // =====================================================
    // 도착 예정 시간
    // =====================================================

    function formatArrival(
        remainingSeconds
    ) {

        const now =
            new Date();


        now.setSeconds(
            now.getSeconds() +
            Math.max(
                0,
                Math.round(
                    remainingSeconds
                )
            )
        );


        return now.toLocaleTimeString(
            "ko-KR",
            {
                hour: "2-digit",
                minute: "2-digit"
            }
        );

    }


    // =====================================================
    // 방향 아이콘
    // =====================================================

    function getTurnIcon(
        guidance
    ) {

        const text =
            String(
                guidance || ""
            );


        if (
            text.includes("유턴")
        ) {

            return "↶";

        }


        if (
            text.includes("좌회전")
        ) {

            return "↰";

        }


        if (
            text.includes("우회전")
        ) {

            return "↱";

        }


        if (
            text.includes("도착")
        ) {

            return "●";

        }


        return "↑";

    }

    // =====================================================
// 다음 안내 팝업
// =====================================================

function showNextGuidePopup(
    guide
) {

    if (
        !guide
    ) {

        return;

    }


    // 기존 팝업 제거
    const oldPopup =
        document.getElementById(
            "next-guide-popup"
        );


    if (oldPopup) {

        oldPopup.remove();

    }


    // 기존 타이머 제거
    if (
        state.guidePopupTimer
    ) {

        clearTimeout(
            state.guidePopupTimer
        );

        state.guidePopupTimer =
            null;

    }


    const guidanceText =
        guide.guidance ||
        guide.name ||
        "";


    const turnIcon =
        getTurnIcon(
            guidanceText
        );


    const turnText =
        getTurnText(
            guidanceText
        );


    const guideDistance =
        Math.max(
            0,
            guide.cumulativeDistance -
            state.traveledMeters
        );


    // =================================================
    // 팝업
    // =================================================

    const popup =
        document.createElement(
            "div"
        );


    popup.id =
        "next-guide-popup";


    popup.innerHTML = `

        <div
            class="next-guide-popup-icon"
        >
            ${turnIcon}
        </div>

        <div
            class="next-guide-popup-distance"
        >
            ${formatDistance(
                guideDistance
            )}
        </div>

        <div
            class="next-guide-popup-text"
        >
            ${turnText}
        </div>

    `;


    // =================================================
    // 팝업 CSS
    // =================================================

    popup.style.position =
        "fixed";

 popup.style.left =
    "50%";

popup.style.top =
    "auto";

popup.style.bottom =
    "105px";

popup.style.transform =
    "translateX(-50%)";

    popup.style.zIndex =
        "99999";

    popup.style.display =
        "flex";

    popup.style.alignItems =
        "center";

    popup.style.justifyContent =
        "center";

  popup.style.gap =
    "21px";

popup.style.padding =
    "27px 42px";

popup.style.background =
    "rgba(90, 90, 90, 0.82)";

    popup.style.borderRadius =
        "12px";

    popup.style.color =
        "#ffffff";

    popup.style.fontFamily =
        "Arial, sans-serif";

    popup.style.pointerEvents =
        "none";


    // =================================================
    // 화살표
    // =================================================

    const icon =
        popup.querySelector(
            ".next-guide-popup-icon"
        );


    icon.style.fontSize =
        "72px";

    icon.style.fontWeight =
        "700";

    icon.style.lineHeight =
        "1";


    // =================================================
    // 거리
    // =================================================

    const distance =
        popup.querySelector(
            ".next-guide-popup-distance"
        );


    distance.style.fontSize =
        "54px";

    distance.style.fontWeight =
        "800";

    distance.style.lineHeight =
        "1";


    // =================================================
    // 안내 문구
    // =================================================

    const text =
        popup.querySelector(
            ".next-guide-popup-text"
        );


    text.style.fontSize =
        "38px";

    text.style.fontWeight =
        "700";

    text.style.whiteSpace =
        "nowrap";


    document.body.appendChild(
        popup
    );


    // =================================================
    // 2초 후 제거
    // =================================================

    state.guidePopupTimer =
        setTimeout(
            function () {

                const currentPopup =
                    document.getElementById(
                        "next-guide-popup"
                    );


                if (
                    currentPopup
                ) {

                    currentPopup.remove();

                }


                state.guidePopupTimer =
                    null;

            },
            2000
        );

}
    // =====================================================
    // 방향 안내 문구
    // =====================================================

    function getTurnText(
        guidance
    ) {

        const text =
            String(
                guidance || ""
            );


        if (
            text.includes("유턴")
        ) {

            return "유턴";

        }


        if (
            text.includes("좌회전")
        ) {

            return "좌회전";

        }


        if (
            text.includes("우회전")
        ) {

            return "우회전";

        }


        if (
            text.includes("도착")
        ) {

            return "도착";

        }


        return "직진";

    }

    // =====================================================
    // 카카오 traffic_state
    //
    // 공식 코드
    // 0 = 정보 없음
    // 1 = 정체
    // 2 = 지체
    // 3 = 서행
    // 4 = 원활
    // 6 = 사고
    // =====================================================

    function trafficLabel(
        stateValue
    ) {

        const value =
            Number(
                stateValue
            );


        switch (value) {

            case 0:
                return "정보 없음";

            case 1:
                return "정체";

            case 2:
                return "지체";

            case 3:
                return "서행";

            case 4:
                return "원활";

            case 6:
                return "사고";

            default:
                return "정보 없음";

        }

    }


    // =====================================================
    // 교통 상태별 색상
    // =====================================================

    function trafficColor(
        stateValue
    ) {

        const value =
            Number(
                stateValue
            );


        switch (value) {

            case 1:
                return "#e74c3c";

            case 2:
                return "#e67e22";

            case 3:
                return "#f1c40f";

            case 4:
                return "#2ecc71";

            case 6:
                return "#8e44ad";

            default:
                return "#7f8c8d";

        }

    }


    // =====================================================
    // 교통속도가 없는 경우의 기본값
    // =====================================================

    function fallbackSpeed(
        trafficState
    ) {

        switch (
            Number(trafficState)
        ) {

            case 1:
                return 15;

            case 2:
                return 25;

            case 3:
                return 35;

            case 4:
                return 50;

            case 6:
                return 10;

            default:
                return 40;

        }

    }


    // =====================================================
    // 배속 UI
    // =====================================================

    function createSpeedControl() {

        if (
            document.getElementById(
                "simulation-speed-control"
            )
        ) {

            return;

        }


        const trafficPanel =
            document.querySelector(
                ".traffic-simulation-panel"
            );


        if (!trafficPanel) {

            console.warn(
                "교통정보 패널을 찾을 수 없습니다."
            );

            return;

        }


        // -------------------------------------------------
        // 스타일
        // -------------------------------------------------

        if (
            !document.getElementById(
                "simulation-speed-control-style"
            )
        ) {

            const style =
                document.createElement(
                    "style"
                );


            style.id =
                "simulation-speed-control-style";


            style.textContent = `

                #simulation-speed-control {

                    margin-top: 12px;
                    padding-top: 10px;

                    border-top:
                        1px solid rgba(
                            255,
                            255,
                            255,
                            0.18
                        );

                }


                #simulation-speed-title {

                    font-size: 12px;
                    font-weight: 600;

                    margin-bottom: 7px;

                    color:
                        rgba(
                            255,
                            255,
                            255,
                            0.82
                        );

                }


                #simulation-speed-buttons {

                    display: flex;

                    gap: 5px;

                    flex-wrap: wrap;

                }


                .simulation-speed-button {

                    border:
                        1px solid
                        rgba(
                            255,
                            255,
                            255,
                            0.35
                        );

                    background:
                        rgba(
                            255,
                            255,
                            255,
                            0.10
                        );

                    color:
                        #ffffff;

                    border-radius:
                        6px;

                    min-width:
                        38px;

                    height:
                        30px;

                    padding:
                        0 7px;

                    font-size:
                        12px;

                    font-weight:
                        700;

                    cursor:
                        pointer;

                    transition:
                        background 0.15s,
                        border-color 0.15s,
                        transform 0.1s;

                }


                .simulation-speed-button:hover {

                    background:
                        rgba(
                            255,
                            255,
                            255,
                            0.20
                        );

                }


                .simulation-speed-button:active {

                    transform:
                        scale(0.96);

                }


                .simulation-speed-button.active {

                    background:
                        #ffffff;

                    color:
                        #1769aa;

                    border-color:
                        #ffffff;

                }


                @media (max-width: 768px) {

                    #simulation-speed-control {

                        margin-top: 8px;
                        padding-top: 7px;

                    }


                    #simulation-speed-title {

                        font-size: 10px;

                    }


                    .simulation-speed-button {

                        min-width:
                            32px;

                        height:
                            27px;

                        padding:
                            0 5px;

                        font-size:
                            10px;

                    }

                }

            `;


            document.head.appendChild(
                style
            );

        }


        // -------------------------------------------------
        // UI 생성
        // -------------------------------------------------

        const control =
            document.createElement(
                "div"
            );


        control.id =
            "simulation-speed-control";


        const title =
            document.createElement(
                "div"
            );


        title.id =
            "simulation-speed-title";


        title.textContent =
            "시뮬레이션 배속";


        const buttons =
            document.createElement(
                "div"
            );


        buttons.id =
            "simulation-speed-buttons";


        const speeds =
            [1, 2, 4, 6, 8, 10];


        speeds.forEach(
            speed => {

                const button =
                    document.createElement(
                        "button"
                    );


                button.type =
                    "button";


                button.className =
                    "simulation-speed-button";


                button.dataset.speed =
                    String(speed);


                button.textContent =
                    `X${speed}`;


                button.addEventListener(
                    "click",
                    () => {

                        setSimulationSpeed(
                            speed
                        );

                    }
                );


                buttons.appendChild(
                    button
                );

            }
        );


        control.appendChild(
            title
        );


        control.appendChild(
            buttons
        );


        trafficPanel.appendChild(
            control
        );


        setSimulationSpeed(
            SIMULATION_MULTIPLIER
        );

    }


    // =====================================================
    // 배속 변경
    // =====================================================

    function setSimulationSpeed(
        speed
    ) {

        const parsedSpeed =
            Number(
                speed
            );


        if (
            !Number.isFinite(
                parsedSpeed
            )
        ) {

            return;

        }


        const safeSpeed =
            Math.min(
                10,
                Math.max(
                    1,
                    Math.round(
                        parsedSpeed
                    )
                )
            );


        SIMULATION_MULTIPLIER =
            safeSpeed;


        const buttons =
            document.querySelectorAll(
                ".simulation-speed-button"
            );


        buttons.forEach(
            button => {

                const buttonSpeed =
                    Number(
                        button.dataset.speed
                    );


                button.classList.toggle(
                    "active",
                    buttonSpeed ===
                    SIMULATION_MULTIPLIER
                );

            }
        );


        setText(
            "simulation-speed-value",
            `X${SIMULATION_MULTIPLIER}`
        );


        console.log(
            `시뮬레이션 배속: X${SIMULATION_MULTIPLIER}`
        );

    }


    // =====================================================
    // 카카오 경로 요청
    // =====================================================

    async function requestRoute(
        origin,
        hospital
    ) {

        const params =
            new URLSearchParams({

                originLat:
                    String(
                        origin.lat
                    ),

                originLng:
                    String(
                        origin.lng
                    ),

                destinationLat:
                    String(
                        hospital.lat
                    ),

                destinationLng:
                    String(
                        hospital.lng
                    )

            });


        const response =
            await fetch(
                `/api/route?${params.toString()}`
            );


        if (
            !response.ok
        ) {

            throw new Error(
                `경로 요청 실패: HTTP ${response.status}`
            );

        }


        const result =
            await response.json();


        if (
            !result.success
        ) {

            throw new Error(
                result.message ||
                "카카오 경로 요청에 실패했습니다."
            );

        }


        const data =
            result.data;


        if (
            !data ||
            !data.routes ||
            !data.routes.length
        ) {

            throw new Error(
                "카카오에서 유효한 경로를 반환하지 않았습니다."
            );

        }


        return data.routes[0];

    }


    // =====================================================
    // 경로 모델 생성
    // =====================================================

    function buildRouteModel(
        route
    ) {

        const geometry = [];

        const roads = [];

        const guides = [];


        let geometryDistance = 0;

        let roadCursor = 0;


        const sections =
            route.sections || [];


        sections.forEach(
            section => {

                const sectionRoads =
                    section.roads || [];


                sectionRoads.forEach(
                    road => {

                        const vertexes =
                            road.vertexes || [];


                        const points = [];


                        for (
                            let i = 0;
                            i < vertexes.length;
                            i += 2
                        ) {

                            const lng =
                                Number(
                                    vertexes[i]
                                );


                            const lat =
                                Number(
                                    vertexes[i + 1]
                                );


                            if (
                                !Number.isFinite(
                                    lat
                                ) ||
                                !Number.isFinite(
                                    lng
                                )
                            ) {

                                continue;

                            }


                            points.push({

                                lat,
                                lng

                            });


                            if (
                                geometry.length
                            ) {

                                const previous =
                                    geometry[
                                        geometry.length - 1
                                    ];


                                geometryDistance +=
                                    distanceMeters(
                                        previous.lat,
                                        previous.lng,
                                        lat,
                                        lng
                                    );

                            }


                            geometry.push({

                                lat,
                                lng

                            });

                        }


                        const roadDistance =
                            Number(
                                road.distance
                            ) || 0;


                        const roadDuration =
                            Number(
                                road.duration
                            ) || 0;


                        const startMeter =
                            roadCursor;


                        roadCursor +=
                            roadDistance;


                        const endMeter =
                            roadCursor;


                        roads.push({

                            name:
                                road.name ||
                                "도로",

                            distance:
                                roadDistance,

                            duration:
                                roadDuration,

                            startMeter,

                            endMeter,

                            trafficSpeed:
                                Number(
                                    road.traffic_speed
                                ) || 0,

                            trafficState:
                                Number(
                                    road.traffic_state
                                ),

                            points

                        });

                    }
                );


                const sectionGuides =
                    section.guides || [];


                sectionGuides.forEach(
                    guide => {

                        guides.push({

                            name:
                                guide.name ||
                                "",

                            guidance:
                                guide.guidance ||
                                "",

                            x:
                                Number(
                                    guide.x
                                ),

                            y:
                                Number(
                                    guide.y
                                ),

                            type:
                                Number(
                                    guide.type
                                ),

                            distance:
                                Number(
                                    guide.distance
                                ) || 0,

                            duration:
                                Number(
                                    guide.duration
                                ) || 0

                        });

                    }
                );

            }
        );


        // -------------------------------------------------
        // guide 누적 거리
        // -------------------------------------------------

        let guideDistance =
            0;


        guides.forEach(
            guide => {

                guideDistance +=
                    guide.distance || 0;


                guide.cumulativeDistance =
                    guideDistance;

            }
        );


        // -------------------------------------------------
        // 전체 거리
        // -------------------------------------------------

        const totalDistance =
            Number(
                route.summary &&
                route.summary.distance
            ) ||
            roadCursor ||
            geometryDistance;


        // -------------------------------------------------
        // 전체 시간
        // -------------------------------------------------

        const totalDuration =
            Number(
                route.summary &&
                route.summary.duration
            ) || 0;


        // =================================================
        // ⭐ 중요
        //
        // 더 이상 signals를 생성하지 않는다.
        // 카카오 guides는 계속 사용하지만
        // 오직 회전/도로 안내용으로만 사용한다.
        // =================================================

        return {

            geometry,

            roads,

            guides,

            totalDistance,

            totalDuration

        };

    }


    // =====================================================
    // 경로 위치 샘플링
    // =====================================================

    function sampleRoute(
        traveledMeters
    ) {

        const points =
            state.model &&
            state.model.geometry;


        if (
            !points ||
            points.length === 0
        ) {

            return null;

        }


        if (
            points.length === 1
        ) {

            return {

                lat:
                    points[0].lat,

                lng:
                    points[0].lng,

                heading:
                    0

            };

        }


        let remaining =
            Math.max(
                0,
                traveledMeters
            );


        for (
            let i = 1;
            i < points.length;
            i++
        ) {

            const previous =
                points[i - 1];


            const current =
                points[i];


            const segmentDistance =
                distanceMeters(
                    previous.lat,
                    previous.lng,
                    current.lat,
                    current.lng
                );


            if (
                remaining <=
                segmentDistance
            ) {

                const ratio =
                    segmentDistance > 0
                        ? remaining /
                          segmentDistance
                        : 0;


                const lat =
                    previous.lat +
                    (
                        current.lat -
                        previous.lat
                    ) *
                    ratio;


                const lng =
                    previous.lng +
                    (
                        current.lng -
                        previous.lng
                    ) *
                    ratio;


                return {

                    lat,

                    lng,

                    heading:
                        bearingDegrees(
                            previous.lat,
                            previous.lng,
                            current.lat,
                            current.lng
                        )

                };

            }


            remaining -=
                segmentDistance;

        }


        const last =
            points[
                points.length - 1
            ];


        const previous =
            points[
                Math.max(
                    0,
                    points.length - 2
                )
            ];


        return {

            lat:
                last.lat,

            lng:
                last.lng,

            heading:
                bearingDegrees(
                    previous.lat,
                    previous.lng,
                    last.lat,
                    last.lng
                )

        };

    }


    // =====================================================
    // 현재 도로
    // =====================================================

    function currentRoad() {

        if (
            !state.model
        ) {

            return null;

        }


        const roads =
            state.model.roads;


        for (
            const road of roads
        ) {

            if (
                state.traveledMeters >=
                road.startMeter &&

                state.traveledMeters <=
                road.endMeter
            ) {

                return road;

            }

        }


        return roads[
            roads.length - 1
        ] || null;

    }


    // =====================================================
    // 다음 안내
    // =====================================================

    function nextGuide() {

        if (
            !state.model
        ) {

            return null;

        }


        for (
            const guide of
            state.model.guides
        ) {

            if (
                guide.cumulativeDistance >
                state.traveledMeters 
            ) {

                return guide;

            }

        }


        return null;

    }


    // =====================================================
    // 목표 속도
    //
    // ⭐ 신호에 의한 감속 없음
    // ⭐ 교통 데이터만 반영
    // =====================================================

    function roadTargetSpeed(
        road
    ) {

        if (
            !road
        ) {

            return 40;

        }


        // 카카오가 현재 교통속도를 제공하면
        // 그대로 사용
        if (
            road.trafficSpeed > 0
        ) {

            return road.trafficSpeed;

        }


        // traffic_speed가 없는 경우
        // traffic_state 기반 fallback
        return fallbackSpeed(
            road.trafficState
        );

    }


    // =====================================================
    // 차량 표시
    // =====================================================

    function updateVehicle(
        position
    ) {

        if (
            !state.map ||
            !position
        ) {

            return;

        }


        if (
            !state.vehicleOverlay
        ) {

            const element =
                document.createElement(
                    "div"
                );


            element.className =
                "traffic-vehicle-marker";


            element.textContent =
                "▲";


            state.vehicleElement =
                element;


            state.vehicleOverlay =
                new kakao.maps.CustomOverlay({

                    position:
                        new kakao.maps.LatLng(
                            position.lat,
                            position.lng
                        ),

                    content:
                        element,

                    yAnchor:
                        0.5,

                    xAnchor:
                        0.5,

                    zIndex:
                        20

                });


            state.vehicleOverlay.setMap(
                state.map
            );

        }


        state.vehicleOverlay.setPosition(
            new kakao.maps.LatLng(
                position.lat,
                position.lng
            )
        );



// -------------------------------------------------
// 차량 방향
// -------------------------------------------------

if (
    state.vehicleElement
) {

    state.vehicleElement.style.transform =
        `rotate(${position.heading}deg)`;

}
   // -------------------------------------------------
// 지도 자동 추적
// 차량을 계속 따라가도록 설정
// -------------------------------------------------

const now =
    performance.now();

if (
    now -
    state.lastMapFollowAt >=
    250
) {

    state.map.setCenter(
        new kakao.maps.LatLng(
            position.lat,
            position.lng
        )
    );

    state.lastMapFollowAt =
        now;

    state.lastMapFollowPosition = {
        lat:
            position.lat,

        lng:
            position.lng
    };
}

    }


    // =====================================================
    // 교통선 제거
    // =====================================================

    function clearTrafficLines() {

        state.trafficLines.forEach(
            line => {

                line.setMap(
                    null
                );

            }
        );


        state.trafficLines =
            [];

    }


    // =====================================================
    // 교통상태별 경로 표시
    // =====================================================

    function drawTrafficRoute() {

        if (
            !state.map ||
            !state.model
        ) {

            return;

        }


        clearTrafficLines();


        state.model.roads.forEach(
            road => {

                if (
                    road.points.length < 2
                ) {

                    return;

                }


                const path =
                    road.points.map(
                        point =>
                            new kakao.maps.LatLng(
                                point.lat,
                                point.lng
                            )
                    );


                const line =
                    new kakao.maps.Polyline({

                        path,

                        strokeWeight:
                            7,

                        strokeColor:
                            trafficColor(
                                road.trafficState
                            ),

                        strokeOpacity:
                            0.85,

                        strokeStyle:
                            "solid"

                    });


                line.setMap(
                    state.map
                );


                state.trafficLines.push(
                    line
                );

            }
        );

    }


    // =====================================================
    // 지도 초기화
    // =====================================================

    function initializeMap() {

        const container =
            document.getElementById(
                "navigation-map"
            );


        if (
            !container
        ) {

            console.error(
                "navigation-map을 찾을 수 없습니다."
            );

            return;

        }


        const firstPoint =
            state.model &&
            state.model.geometry &&
            state.model.geometry[0];


        if (
            !firstPoint
        ) {

            return;

        }


        state.map =
            new kakao.maps.Map(
                container,
                {

                    center:
                        new kakao.maps.LatLng(
                            firstPoint.lat,
                            firstPoint.lng
                        ),

                    level:
                        1

                }
            );


        // -------------------------------------------------
        // 병원 마커
        // -------------------------------------------------

        if (
            state.hospital
        ) {

            state.hospitalMarker =
                new kakao.maps.Marker({

                    position:
                        new kakao.maps.LatLng(
                            state.hospital.lat,
                            state.hospital.lng
                        ),

                    map:
                        state.map

                });

        }


        // -------------------------------------------------
        // 교통 경로
        // -------------------------------------------------

        drawTrafficRoute();


        // -------------------------------------------------
        // 응급차량
        // -------------------------------------------------

        updateVehicle(
            firstPoint
        );


        // -------------------------------------------------
        // 배속 UI
        // -------------------------------------------------

        createSpeedControl();


        setSimulationSpeed(
            SIMULATION_MULTIPLIER
        );


        // -------------------------------------------------
        // 신호 UI 제거
        // -------------------------------------------------

        removeTrafficSignalUI();

    }


    // =====================================================
    // 화면 업데이트
    // =====================================================

    function updateDisplay() {

        if (
            !state.model
        ) {

            return;

        }


        // -------------------------------------------------
        // 남은 거리
        // -------------------------------------------------

        const remainingDistance =
            Math.max(
                0,
                state.model.totalDistance -
                state.traveledMeters
            );


        const currentRoadInfo =
            currentRoad();


        const targetSpeed =
            roadTargetSpeed(
                currentRoadInfo
            );


        // -------------------------------------------------
        // 현재 속도
        // -------------------------------------------------

        setText(
            "current-speed",
            `${Math.round(
                state.currentSpeedKmh
            )}`
        );


        // -------------------------------------------------
        // 남은 거리
        // -------------------------------------------------

        setText(
            "navigation-remaining-distance",
            formatDistance(
                remainingDistance
            )
        );


        // -------------------------------------------------
        // ETA
        // -------------------------------------------------

        const remainingSeconds =
            targetSpeed > 0
                ? (
                    remainingDistance /
                    1000
                ) /
                targetSpeed *
                3600
                : 0;


        setText(
            "navigation-arrival-time",
            formatArrival(
                remainingSeconds
            )
        );


        // -------------------------------------------------
        // 교통상태
        // -------------------------------------------------

        if (
            currentRoadInfo
        ) {

            setText(
                "traffic-state",
                trafficLabel(
                    currentRoadInfo.trafficState
                )
            );


            setText(
                "traffic-target-speed",
                `${Math.round(
                    targetSpeed
                )} km/h`
            );

        }


        // -------------------------------------------------
        // 배속
        // -------------------------------------------------

        setText(
            "simulation-speed-value",
            `X${SIMULATION_MULTIPLIER}`
        );


        // -------------------------------------------------
        // 다음 안내
        // -------------------------------------------------

        const guide =
            nextGuide();


        if (
            guide
        ) {

            const guideDistance =
                Math.max(
                    0,
                    guide.cumulativeDistance -
                    state.traveledMeters
                );


            const guidanceText =
                guide.guidance ||
                guide.name ||
                "";

                // -------------------------------------------------
// 다음 안내 변경 감지
// -------------------------------------------------

const guideKey =
    `${guide.cumulativeDistance}|${guidanceText}`;


if (
    state.currentGuideKey !== null &&
    state.currentGuideKey !== guideKey
) {

    showNextGuidePopup(
        guide
    );

}


state.currentGuideKey =
    guideKey;


            setText(
                "navigation-direction-icon",
                getTurnIcon(
                    guidanceText
                )
            );


            setText(
                "navigation-distance",
                formatDistance(
                    guideDistance
                )
            );


                        setText(
                "navigation-road-name",
                getTurnText(
                    guidanceText
                )
            );


            setText(
                "next-guide-distance",
                formatDistance(
                    guideDistance
                )
            );


            setText(
                "traffic-sign",
                guidanceText ||
                guide.name ||
                "경로 안내"
            );

        }
        else {

            setText(
                "navigation-direction-icon",
                "●"
            );


            setText(
                "navigation-distance",
                "도착"
            );


            setText(
                "navigation-road-name",
                state.hospital &&
                state.hospital.name
                    ? state.hospital.name
                    : "목적지"
            );


            setText(
                "next-guide-distance",
                "도착"
            );


            setText(
                "traffic-sign",
                "목적지 도착"
            );

        }


        // -------------------------------------------------
        // 신호 UI는 항상 숨김
        // -------------------------------------------------

        removeTrafficSignalUI();


        // -------------------------------------------------
        // 교통정보 갱신 시간
        // -------------------------------------------------

        if (
            state.lastRefreshAt
        ) {

            const elapsed =
                Math.floor(
                    (
                        Date.now() -
                        state.lastRefreshAt
                    ) / 1000
                );


            setText(
                "traffic-refresh-time",
                `${elapsed}초 전`
            );

        }

    }


    // =====================================================
    // 가속 / 감속
    // =====================================================

    function accelerateToward(
        targetSpeed,
        deltaSeconds
    ) {

        const acceleration =
            8;


        const difference =
            targetSpeed -
            state.currentSpeedKmh;


        const maxChange =
            acceleration *
            deltaSeconds;


        if (
            Math.abs(
                difference
            ) <=
            maxChange
        ) {

            state.currentSpeedKmh =
                targetSpeed;

        }
        else {

            state.currentSpeedKmh +=
                Math.sign(
                    difference
                ) *
                maxChange;

        }


        state.currentSpeedKmh =
            Math.max(
                0,
                state.currentSpeedKmh
            );

    }


    // =====================================================
    // ⭐ 교통정보 재조회
    //
    // 60초마다 실행
    // =====================================================

    async function refreshTrafficRoute() {

        if (
            !state.active ||
            state.refreshing ||
            !state.hospital ||
            !state.currentPosition
        ) {

            return;

        }


        state.refreshing =
            true;


        try {

            console.log(
                "60초 경과 → 현재 위치에서 교통정보 재조회"
            );


            const newRoute =
                await requestRoute(
                    state.currentPosition,
                    state.hospital
                );


            const newModel =
                buildRouteModel(
                    newRoute
                );


            if (
                !newModel.geometry.length
            ) {

                throw new Error(
                    "새로운 경로의 도로 좌표가 없습니다."
                );

            }


            state.route =
                newRoute;


            state.model =
                newModel;


            // -------------------------------------------------
            // 새 경로는 현재 응급차 위치에서 시작
            // -------------------------------------------------

            state.traveledMeters = 0;

            state.currentGuideKey =
    null;


if (
    state.guidePopupTimer
) {

    clearTimeout(
        state.guidePopupTimer
    );

    state.guidePopupTimer =
        null;

}


            state.lastRefreshAt =
                Date.now();

                state.lastRefreshAt =
    Date.now();

state.lastMapFollowAt =
    0;

state.lastMapFollowPosition =
    null;


            // -------------------------------------------------
            // 새 경로의 첫 위치를 현재 위치로 설정
            // -------------------------------------------------

            state.currentPosition = {

                lat:
                    newModel.geometry[0].lat,

                lng:
                    newModel.geometry[0].lng

            };


            state.currentHeading =
                0;


            drawTrafficRoute();


            updateDisplay();


            console.log(
                "교통정보 및 경로 갱신 완료"
            );

        }
        catch (error) {

            console.error(
                "교통정보 재조회 실패:",
                error
            );

        }
        finally {

            state.refreshing =
                false;

        }

    }

    // =====================================================
// 경로 재탐색 팝업
// =====================================================

function showReroutePopup() {

    const oldPopup =
        document.getElementById(
            "reroute-popup"
        );


    if (
        oldPopup
    ) {

        oldPopup.remove();

    }


    const popup =
        document.createElement(
            "div"
        );


    popup.id =
        "reroute-popup";


    popup.textContent =
        "경로 재탐색";


    popup.style.position =
        "fixed";

    popup.style.left =
        "50%";

    popup.style.top =
        "50%";

    popup.style.transform =
        "translate(-50%, -50%)";

    popup.style.zIndex =
        "99999";

    popup.style.padding =
        "20px 32px";

    popup.style.background =
        "rgba(90, 90, 90, 0.82)";

    popup.style.color =
        "#ffffff";

    popup.style.borderRadius =
        "12px";

    popup.style.fontSize =
        "28px";

    popup.style.fontWeight =
        "700";

    popup.style.fontFamily =
        "Arial, sans-serif";

    popup.style.whiteSpace =
        "nowrap";

    popup.style.pointerEvents =
        "none";

    popup.style.boxShadow =
        "0 4px 16px rgba(0, 0, 0, 0.3)";


    document.body.appendChild(
        popup
    );


    return popup;

}

// =====================================================
// 수동 경로 재탐색
// =====================================================

async function manualReroute() {

    if (
        !state.active ||
        state.refreshing ||
        !state.hospital ||
        !state.currentPosition
    ) {

        return;

    }


    const popup =
        showReroutePopup();


    try {

        await refreshTrafficRoute();

    }

    catch (error) {

        console.error(
            "수동 경로 재탐색 실패:",
            error
        );

    }


    state.reroutePopupTimer =
    setTimeout(
        () => {

            if (
                popup &&
                popup.parentNode
            ) {

                popup.remove();

            }


            state.reroutePopupTimer =
                null;

        },
        1000
    );
}


    // =====================================================
    // 주행 종료
    // =====================================================

    function finishNavigation() {

        state.active =
            false;


        if (
            state.timerId
        ) {

            clearTimeout(
                state.timerId
            );


            state.timerId =
                null;

        }


        state.currentSpeedKmh =
            0;


        setText(
            "current-speed",
            "0"
        );


        setText(
            "navigation-direction-icon",
            "●"
        );


        setText(
            "navigation-distance",
            "도착"
        );


        setText(
            "navigation-road-name",
            state.hospital &&
            state.hospital.name
                ? state.hospital.name
                : "목적지 도착"
        );


        setText(
            "navigation-remaining-distance",
            "0 m"
        );


        setText(
            "traffic-state",
            "도착"
        );


        setText(
            "traffic-target-speed",
            "0 km/h"
        );


        setText(
            "traffic-sign",
            "목적지 도착"
        );


        removeTrafficSignalUI();


        console.log(
            "응급차량이 목적지에 도착했습니다."
        );

    }


    // =====================================================
    // Tick
    // =====================================================

    function tick() {

        if (
            !state.active
        ) {

            return;

        }


        const now =
            performance.now();


        if (
            !state.lastTickAt
        ) {

            state.lastTickAt =
                now;

        }


        const realDeltaSeconds =
            Math.min(
                1,
                Math.max(
                    0,
                    (
                        now -
                        state.lastTickAt
                    ) / 1000
                )
            );


        state.lastTickAt =
            now;


        // -------------------------------------------------
        // 실제 시간 × 시뮬레이션 배속
        // -------------------------------------------------

        const simulationDeltaSeconds =
            realDeltaSeconds *
            SIMULATION_MULTIPLIER;


        state.simulationClockSeconds +=
            simulationDeltaSeconds;


        // -------------------------------------------------
        // 현재 도로의 교통상황만 확인
        // -------------------------------------------------

        const road =
            currentRoad();


        const targetSpeed =
            roadTargetSpeed(
                road
            );


        // -------------------------------------------------
        // ⭐ 신호 관련 로직 없음
        //
        // 빨간불
        // 초록불
        // 신호 거리
        // 신호 감속
        //
        // 모두 사용하지 않음
        // -------------------------------------------------


        // -------------------------------------------------
        // 가속 / 감속
        //
        // 오직 도로 교통속도에 따라 결정
        // -------------------------------------------------

        accelerateToward(
            targetSpeed,
            simulationDeltaSeconds
        );


        // -------------------------------------------------
        // 이동
        // -------------------------------------------------

        const movementMeters =
            (
                state.currentSpeedKmh /
                3.6
            ) *
            simulationDeltaSeconds;


        state.traveledMeters +=
            movementMeters;


        // -------------------------------------------------
        // 현재 위치
        // -------------------------------------------------

        const position =
            sampleRoute(
                state.traveledMeters
            );


        if (
            position
        ) {

            state.currentPosition =
                position;


            state.currentHeading =
                position.heading;


            updateVehicle(
                position
            );

        }


        // -------------------------------------------------
        // 화면 업데이트
        // -------------------------------------------------

        updateDisplay();


        // -------------------------------------------------
        // 실제 시간 기준 교통 경로 자동 갱신
        // -------------------------------------------------

        if (
            state.lastRefreshAt &&
            Date.now() - state.lastRefreshAt >=
                TRAFFIC_REFRESH_INTERVAL_MS
        ) {

            void refreshTrafficRoute();

        }


        // -------------------------------------------------
        // 도착
        // -------------------------------------------------

        const remainingDistance =
            state.model
                ? state.model.totalDistance -
                  state.traveledMeters
                : Infinity;


        if (
            remainingDistance <= 10 ||
            state.traveledMeters >=
            state.model.totalDistance *
            0.999
        ) {

            finishNavigation();

            return;

        }


        // -------------------------------------------------
        // 다음 Tick
        // -------------------------------------------------

        state.timerId =
            setTimeout(
                tick,
                TICK_INTERVAL_MS
            );

    }


    // =====================================================
    // 타이머 시작
    // =====================================================

    function startTimer() {

        if (
            state.timerId
        ) {

            clearTimeout(
                state.timerId
            );

        }


        state.lastTickAt =
            performance.now();


        state.timerId =
            setTimeout(
                tick,
                TICK_INTERVAL_MS
            );

    }


    // =====================================================
    // 지도 객체 제거
    // =====================================================

    function clearMapObjects() {

        clearTrafficLines();


        if (
            state.hospitalMarker
        ) {

            state.hospitalMarker.setMap(
                null
            );


            state.hospitalMarker =
                null;

        }


        if (
            state.vehicleOverlay
        ) {

            state.vehicleOverlay.setMap(
                null
            );


            state.vehicleOverlay =
                null;


            state.vehicleElement =
                null;

        }


        state.map =
            null;

    }


    // =====================================================
    // ⭐ 네비게이션 시작
    // =====================================================

    async function startNavigation(
        hospital
    ) {

        try {

            if (
                !hospital
            ) {

                alert(
                    "선택한 병원 정보가 없습니다."
                );

                return;

            }


            const getPatientLocation =
                window.getPatientLocation;


            if (
                typeof getPatientLocation !==
                "function"
            ) {

                alert(
                    "환자 위치 정보를 가져올 수 없습니다."
                );

                return;

            }


            const patientLocation =
                getPatientLocation();


            if (
                !patientLocation ||
                !Number.isFinite(
                    Number(
                        patientLocation.lat
                    )
                ) ||
                !Number.isFinite(
                    Number(
                        patientLocation.lng
                    )
                )
            ) {

                alert(
                    "먼저 환자 위치를 생성해주세요."
                );

                return;

            }


            // -------------------------------------------------
            // 기존 주행 종료
            // -------------------------------------------------

            if (
                state.timerId
            ) {

                clearTimeout(
                    state.timerId
                );


                state.timerId =
                    null;

            }


            state.active =
                false;


            clearMapObjects();


            // -------------------------------------------------
            // 기본 배속 X1
            // -------------------------------------------------

            SIMULATION_MULTIPLIER =
                1;


            // -------------------------------------------------
            // 신호 UI 제거
            // -------------------------------------------------

            removeTrafficSignalUI();


            // -------------------------------------------------
            // 경로 계산
            // -------------------------------------------------

            setText(
                "traffic-state",
                "경로 계산 중..."
            );


            const route =
                await requestRoute(
                    patientLocation,
                    hospital
                );


            const model =
                buildRouteModel(
                    route
                );


            if (
                !model.geometry.length
            ) {

                throw new Error(
                    "주행할 수 있는 도로 좌표가 없습니다."
                );

            }


            // -------------------------------------------------
            // 상태 초기화
            // -------------------------------------------------

            state.hospital =
                hospital;


            state.route =
                route;


            state.model =
                model;


            state.traveledMeters =
                0;


            state.simulationClockSeconds =
                0;


            state.currentSpeedKmh =
                0;


            state.currentPosition = {

                lat:
                    model.geometry[0].lat,

                lng:
                    model.geometry[0].lng

            };


            state.currentHeading =
                0;


            state.refreshing =
                false;


            state.lastRefreshAt =
                Date.now();


            state.active =
                true;


            // -------------------------------------------------
            // 네비게이션 화면
            // -------------------------------------------------

            document.body.classList.add(
                "navigation-mode"
            );


            // -------------------------------------------------
            // 배속 UI
            // -------------------------------------------------

            setTimeout(
                () => {

                    createSpeedControl();


                    setSimulationSpeed(
                        1
                    );


                    removeTrafficSignalUI();

                },
                100
            );


            // -------------------------------------------------
            // 지도 초기화
            // -------------------------------------------------

            setTimeout(
                () => {

                    initializeMap();


                    updateDisplay();


                    startTimer();

                },
                150
            );


            console.log(
                "================================="
            );


            console.log(
                "응급차량 네비게이션 시작"
            );


            console.log(
                "병원:",
                hospital.name
            );


            console.log(
                "시뮬레이션 배속:",
                "X1"
            );


            console.log(
                "교통정보 재조회:",
                "60초"
            );


            console.log(
                "신호등:",
                "사용하지 않음"
            );


            console.log(
                "================================="
            );

        }
        catch (error) {

            console.error(
                "네비게이션 시작 오류:",
                error
            );


            state.active =
                false;


            alert(
                error.message ||
                "네비게이션을 시작할 수 없습니다."
            );

        }

    }


    // =====================================================
    // 현재 위치로 지도 이동
    // =====================================================

    function recenterNavigationMap() {

        if (
            !state.map ||
            !state.currentPosition
        ) {

            return;

        }


        state.map.setCenter(
            new kakao.maps.LatLng(
                state.currentPosition.lat,
                state.currentPosition.lng
            )
        );


        state.map.setLevel(
            1
        );

    }


    // =====================================================
    // 네비게이션 종료
    // =====================================================

    function exitNavigation() {

        state.active =
            false;


        if (
            state.timerId
        ) {

            clearTimeout(
                state.timerId
            );


            state.timerId =
                null;

        }


        clearMapObjects();


        state.hospital =
            null;


        state.route =
            null;


        state.model =
            null;


        state.currentPosition =
            null;


        state.traveledMeters =
            0;


        state.currentSpeedKmh =
            0;


        document.body.classList.remove(
            "navigation-mode"
        );


        SIMULATION_MULTIPLIER =
            1;


        removeTrafficSignalUI();


        console.log(
            "네비게이션 종료"
        );

    }


    // =====================================================
    // 외부 공개
    // =====================================================

    window.startNavigation =
        startNavigation;


    window.exitNavigation =
        exitNavigation;


    window.recenterNavigationMap =
        recenterNavigationMap;

    window.manualReroute =
        manualReroute;


    window.setSimulationSpeed =
        setSimulationSpeed;


    window.getSimulationSpeed =
        function () {

            return SIMULATION_MULTIPLIER;

        };


    window.isTrafficSimulationNavigation =
        function () {

            return state.active;

        };


    window.getTrafficSimulationState =
        function () {

            return state;

        };


})();
