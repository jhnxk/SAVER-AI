// =========================================================
// SAVER-AI 의료 자연어 보수적 후처리
//
// 원칙
// - STT에 실제로 존재하는 표현만 정리한다.
// - 누락된 증상/나이/성별/좌우를 추측해서 복원하지 않는다.
// - 말더듬은 filler와 연속 중복 표현만 보수적으로 제거한다.
// - 활력징후처럼 문맥이 명확한 숫자/단위만 표준 표기로 바꾼다.
// =========================================================

(() => {
  "use strict";

  const SINO_DIGITS = {
    영: 0,
    공: 0,
    일: 1,
    이: 2,
    삼: 3,
    사: 4,
    오: 5,
    육: 6,
    칠: 7,
    팔: 8,
    구: 9,
  };

  const SINO_UNITS = {
    십: 10,
    백: 100,
    천: 1000,
    만: 10000,
  };

  const NATIVE_ONES = {
    하나: 1,
    한: 1,
    둘: 2,
    두: 2,
    셋: 3,
    세: 3,
    넷: 4,
    네: 4,
    다섯: 5,
    여섯: 6,
    일곱: 7,
    여덟: 8,
    아홉: 9,
  };

  const NATIVE_TENS = {
    열: 10,
    스물: 20,
    스무: 20,
    서른: 30,
    마흔: 40,
    쉰: 50,
    예순: 60,
    일흔: 70,
    여든: 80,
    아흔: 90,
  };

  const NUMBER_ATOM =
    "(?:\\d+(?:\\.\\d+)?|[영공일이삼사오육칠팔구십백천만점]+|(?:(?:하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열|스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔)\\s*)+)";

  const SUBJECT_PARTICLE = "(?:은|는|이|가)?";

  const PROTECTED_DUPLICATES = new Set([
    "아니",
    "아니고",
    "없음",
    "없습니다",
    "있음",
    "있습니다",
    "왼쪽",
    "오른쪽",
    "좌측",
    "우측",
  ]);

  const SELF_CORRECTION_WORDS = new Set([
    "아니",
    "아니고",
    "정정",
  ]);

  function cleanText(value) {
    return String(value || "")
      .replace(/\s+/g, " ")
      .replace(/\s+([,.!?])/g, "$1")
      .trim();
  }

  function parseSinoInteger(raw) {
    const text = String(raw || "").replace(/\s+/g, "");
    if (!text) return null;

    let total = 0;
    let section = 0;
    let number = 0;
    let saw = false;

    for (const ch of text) {
      if (Object.prototype.hasOwnProperty.call(SINO_DIGITS, ch)) {
        number = SINO_DIGITS[ch];
        saw = true;
        continue;
      }

      const unit = SINO_UNITS[ch];
      if (!unit) return null;

      saw = true;

      if (unit === 10000) {
        section += number;
        if (section === 0) section = 1;
        total += section * 10000;
        section = 0;
        number = 0;
      } else {
        if (number === 0) number = 1;
        section += number * unit;
        number = 0;
      }
    }

    if (!saw) return null;
    return total + section + number;
  }

  function parseNativeInteger(raw) {
    const text = String(raw || "").replace(/\s+/g, "");
    if (!text) return null;

    const tokens = [
      ...Object.keys(NATIVE_TENS),
      ...Object.keys(NATIVE_ONES),
    ].sort((a, b) => b.length - a.length);

    let cursor = 0;
    let total = 0;
    let saw = false;

    while (cursor < text.length) {
      const token = tokens.find((candidate) =>
        text.startsWith(candidate, cursor)
      );

      if (!token) return null;

      if (Object.prototype.hasOwnProperty.call(NATIVE_TENS, token)) {
        total += NATIVE_TENS[token];
      } else {
        total += NATIVE_ONES[token];
      }

      cursor += token.length;
      saw = true;
    }

    return saw ? total : null;
  }

  function parseKoreanNumber(raw) {
    const original = cleanText(raw);
    if (!original) return null;

    const compact = original.replace(/\s+/g, "");

    if (/^\d+(?:\.\d+)?$/.test(compact)) {
      const n = Number(compact);
      return Number.isFinite(n) ? n : null;
    }

    if (compact.includes("점")) {
      const [leftRaw, ...rightParts] = compact.split("점");
      const rightRaw = rightParts.join("");
      const left = parseKoreanNumber(leftRaw);
      if (left === null || !rightRaw) return null;

      let decimals = "";

      for (const ch of rightRaw) {
        if (Object.prototype.hasOwnProperty.call(SINO_DIGITS, ch)) {
          decimals += String(SINO_DIGITS[ch]);
        } else if (/\d/.test(ch)) {
          decimals += ch;
        } else {
          return null;
        }
      }

      if (!decimals) return null;
      return Number(`${left}.${decimals}`);
    }

    const sino = parseSinoInteger(compact);
    if (sino !== null) return sino;

    return parseNativeInteger(compact);
  }

  function formatNumber(value) {
    if (
      value === null ||
      value === undefined ||
      !Number.isFinite(value)
    ) {
      return null;
    }

    return String(value);
  }

  function createTracker(rawText) {
    const corrections = [];
    const warnings = [];
    let text = rawText;

    function replace(regex, replacer, type) {
      text = text.replace(regex, (...args) => {
        const full = args[0];
        const replaced = replacer(...args);

        if (
          typeof replaced !== "string" ||
          replaced === full
        ) {
          return full;
        }

        corrections.push({
          type,
          from: full,
          to: replaced,
        });

        return replaced;
      });
    }

    function warn(regex, message) {
      if (regex.test(text)) {
        warnings.push(message);
      }
    }

    return {
      get text() {
        return text;
      },

      set text(value) {
        text = value;
      },

      corrections,
      warnings,
      replace,
      warn,
    };
  }

  function normalizeLexicalErrors(tracker) {
    const rules = [
      [
        /시티(?=\s*(?:검사|촬영|와|MRI|가|를|는|이|$))/gi,
        "CT",
      ],
      [
        /에스\s*피\s*오\s*투/gi,
        "SpO2",
      ],
      [
        /에스피오투/gi,
        "SpO2",
      ],
      [
        /에스\s*티\s*상승/gi,
        "ST 상승",
      ],
      [
        /소화\s*응급\s*진료/g,
        "소아 응급 진료",
      ],
      [
        /수익\s*치료/g,
        "수액 치료",
      ],
      [
        /전자반증/g,
        "전자간증",
      ],
      [
        /산소\s*포화도/g,
        "산소포화도",
      ],
      [
        /시먼\s*두통/g,
        "심한 두통",
      ],
    ];

    for (const [regex, replacement] of rules) {
      tracker.replace(
        regex,
        () => replacement,
        "medical_term"
      );
    }
  }

  function normalizeDisfluency(tracker) {
    tracker.replace(
      /(^|[\s,])(?:(?:어|음|저기)(?:[,.…]*\s+)){1,4}/g,
      (_full, prefix) => prefix,
      "filler"
    );

    tracker.replace(
      /(^|\s)([가-힣])(?:[,.…-]*\s+)(\2[가-힣]{1,})(?=\s|$|[,.!?])/g,
      (
        full,
        prefix,
        _syllable,
        word
      ) => {
        if (
          SELF_CORRECTION_WORDS.has(
            word
          )
        ) {
          return full;
        }

        return `${prefix}${word}`;
      },
      "stutter"
    );

    tracker.replace(
      /(^|\s)([가-힣A-Za-z]{2,})(?:(?:\s*[,.…]+\s*|\s+)\2){1,3}(?=\s|$|[,.!?])/g,
      (
        full,
        prefix,
        token
      ) => {
        if (
          PROTECTED_DUPLICATES.has(
            token
          )
        ) {
          return full;
        }

        return `${prefix}${token}`;
      },
      "repetition"
    );

    tracker.warn(
      /(?:아니|정정|아니고|말고)/,
      "자기수정 표현이 포함되어 있습니다. 좌우·숫자·나이·성별 같은 핵심 정보는 화면에서 한 번 확인해 주세요."
    );
  }

  function normalizeAge(tracker) {
    const ageRegex =
      new RegExp(
        `(${NUMBER_ATOM})\\s*(?:세|살)(?=\\s|,|$)`,
        "g"
      );

    tracker.replace(
      ageRegex,
      (
        full,
        rawNumber
      ) => {
        const value =
          parseKoreanNumber(
            rawNumber
          );

        if (
          value === null ||
          value < 0 ||
          value > 130
        ) {
          return full;
        }

        return `${formatNumber(
          value
        )}세`;
      },
      "age"
    );
  }

  function normalizeBloodPressure(tracker) {
    const bpRegex =
      new RegExp(
        `혈압${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:에|대|/)\\s*(${NUMBER_ATOM})(?:\\s*(?:mmhg|mmHg|밀리미터에이치지))?`,
        "gi"
      );

    tracker.replace(
      bpRegex,
      (
        full,
        systolicRaw,
        diastolicRaw
      ) => {
        const systolic =
          parseKoreanNumber(
            systolicRaw
          );

        const diastolic =
          parseKoreanNumber(
            diastolicRaw
          );

        if (
          systolic === null ||
          diastolic === null
        ) {
          return full;
        }

        if (
          systolic < 20 ||
          systolic > 350 ||
          diastolic < 10 ||
          diastolic > 250
        ) {
          return full;
        }

        return (
          `혈압 ` +
          `${formatNumber(
            systolic
          )}/` +
          `${formatNumber(
            diastolic
          )}`
        );
      },
      "blood_pressure"
    );

    const explicitBpRegex =
      new RegExp(
        `수축기(?:\\s*혈압)?${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:,|이고|이며|\\s)+이완기(?:\\s*혈압)?${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})`,
        "g"
      );

    tracker.replace(
      explicitBpRegex,
      (
        full,
        systolicRaw,
        diastolicRaw
      ) => {
        const systolic =
          parseKoreanNumber(
            systolicRaw
          );

        const diastolic =
          parseKoreanNumber(
            diastolicRaw
          );

        if (
          systolic === null ||
          diastolic === null
        ) {
          return full;
        }

        return (
          `혈압 ` +
          `${formatNumber(
            systolic
          )}/` +
          `${formatNumber(
            diastolic
          )}`
        );
      },
      "blood_pressure"
    );
  }

  function normalizeOxygenSaturation(tracker) {
    const spo2Regex =
      new RegExp(
        `(?:산소포화도|SpO2)${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:퍼센트|프로|%)`,
        "gi"
      );

    tracker.replace(
      spo2Regex,
      (
        full,
        numberRaw
      ) => {
        const value =
          parseKoreanNumber(
            numberRaw
          );

        if (
          value === null ||
          value < 0 ||
          value > 100
        ) {
          return full;
        }

        return `SpO2 ${formatNumber(
          value
        )}%`;
      },
      "spo2"
    );
  }

  function normalizeTemperature(tracker) {
    const tempRegex =
      new RegExp(
        `체온${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:도씨|도|℃)`,
        "g"
      );

    tracker.replace(
      tempRegex,
      (
        full,
        numberRaw
      ) => {
        const value =
          parseKoreanNumber(
            numberRaw
          );

        if (
          value === null ||
          value < 20 ||
          value > 50
        ) {
          return full;
        }

        return `체온 ${formatNumber(
          value
        )}℃`;
      },
      "temperature"
    );
  }

  function normalizeRate(
    tracker,
    label
  ) {
    const rateRegex =
      new RegExp(
        `${label}${SUBJECT_PARTICLE}\\s*(?:분당\\s*)?(${NUMBER_ATOM})\\s*(?:회|번)(?:\\s*(?:/\\s*분|퍼\\s*분|분당))?`,
        "g"
      );

    tracker.replace(
      rateRegex,
      (
        full,
        numberRaw
      ) => {
        const value =
          parseKoreanNumber(
            numberRaw
          );

        if (
          value === null ||
          value < 0 ||
          value > 400
        ) {
          return full;
        }

        const hasPerMinute =
          /분당|퍼\s*분|\/\s*분/.test(
            full
          );

        return (
          `${label} ` +
          `${formatNumber(
            value
          )}회` +
          `${hasPerMinute
            ? "/분"
            : ""}`
        );
      },
      label === "맥박"
        ? "pulse"
        : "respiratory_rate"
    );
  }

  function normalizeBodyMeasurements(
    tracker
  ) {
    const weightRegex =
      new RegExp(
        `체중${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:킬로그램|킬로|kg)`,
        "gi"
      );

    tracker.replace(
      weightRegex,
      (
        full,
        raw
      ) => {
        const value =
          parseKoreanNumber(raw);

        if (
          value === null ||
          value <= 0 ||
          value > 500
        ) {
          return full;
        }

        return `체중 ${formatNumber(
          value
        )}kg`;
      },
      "weight"
    );

    const heightRegex =
      new RegExp(
        `(?:키|신장)${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:센티미터|센티|cm)`,
        "gi"
      );

    tracker.replace(
      heightRegex,
      (
        full,
        raw
      ) => {
        const value =
          parseKoreanNumber(raw);

        if (
          value === null ||
          value <= 0 ||
          value > 250
        ) {
          return full;
        }

        return `키 ${formatNumber(
          value
        )}cm`;
      },
      "height"
    );
  }

  function normalizeOxygenFlow(
    tracker
  ) {
    const oxygenRegex =
      new RegExp(
        `산소${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:리터|L)(?:\\s*(?:퍼\\s*분|분당|/\\s*분))?`,
        "gi"
      );

    tracker.replace(
      oxygenRegex,
      (
        full,
        raw
      ) => {
        const value =
          parseKoreanNumber(raw);

        if (
          value === null ||
          value < 0 ||
          value > 60
        ) {
          return full;
        }

        const perMinute =
          /퍼\s*분|분당|\/\s*분/.test(
            full
          );

        return (
          `산소 ` +
          `${formatNumber(
            value
          )}L` +
          `${perMinute
            ? "/min"
            : ""}`
        );
      },
      "oxygen_flow"
    );
  }

  function normalizeBloodGlucose(
    tracker
  ) {
    const glucoseRegex =
      new RegExp(
        `혈당${SUBJECT_PARTICLE}\\s*(${NUMBER_ATOM})\\s*(?:mg\\s*/\\s*dL|밀리그램\\s*(?:퍼|매)\\s*데시리터)`,
        "gi"
      );

    tracker.replace(
      glucoseRegex,
      (
        full,
        raw
      ) => {
        const value =
          parseKoreanNumber(raw);

        if (
          value === null ||
          value < 0 ||
          value > 2000
        ) {
          return full;
        }

        return `혈당 ${formatNumber(
          value
        )}mg/dL`;
      },
      "glucose"
    );
  }

  function normalize(rawText) {
    const raw =
      cleanText(rawText);

    const tracker =
      createTracker(raw);

    normalizeLexicalErrors(
      tracker
    );

    normalizeDisfluency(
      tracker
    );

    normalizeAge(
      tracker
    );

    normalizeBloodPressure(
      tracker
    );

    normalizeOxygenSaturation(
      tracker
    );

    normalizeTemperature(
      tracker
    );

    normalizeRate(
      tracker,
      "맥박"
    );

    normalizeRate(
      tracker,
      "호흡수"
    );

    normalizeBodyMeasurements(
      tracker
    );

    normalizeOxygenFlow(
      tracker
    );

    normalizeBloodGlucose(
      tracker
    );

    tracker.text =
      cleanText(
        tracker.text
      );

    return {
      rawText: raw,
      normalizedText:
        tracker.text,
      corrections:
        tracker.corrections,
      correctionCount:
        tracker.corrections.length,
      warnings:
        tracker.warnings,
      warningCount:
        tracker.warnings.length,
    };
  }

  window.SAVERMedicalTextNormalizer = {
    normalize,
    parseKoreanNumber,
  };
})();