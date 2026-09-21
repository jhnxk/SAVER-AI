// =========================================================
// SAVER-AI 실시간 Google Cloud Speech-to-Text 음성 입력
//
// 흐름
// 마이크 PCM16 → WebSocket → Flask → Google StreamingRecognize
// → interim 결과를 즉시 textarea에 표시
// → final 결과는 의료 자연어 후처리 후 확정
//
// 기존 Gemini / 병원 추천 / DB / GPS / 지도 로직은 건드리지 않는다.
// =========================================================

(() => {
  "use strict";

  const MAX_RECORDING_MS = 30 * 1000;
  const PCM_CHUNK_SAMPLES = 4096;
  const STOP_GRACE_MS = 5000;

  let mediaStream = null;
  let audioContext = null;
  let sourceNode = null;
  let captureNode = null;
  let silentGainNode = null;
  let workletBlobUrl = null;
  let webSocket = null;
  let autoStopTimer = null;
  let stopGraceTimer = null;

  let isConnecting = false;
  let isRecording = false;
  let isStopping = false;
  let serverFinished = false;
  let manualStopRequested = false;
  let fatalHandled = false;

  let baseText = "";
  let finalTranscript = "";
  let interimTranscript = "";

  let pcmBuffer = new Float32Array(PCM_CHUNK_SAMPLES);
  let pcmBufferLength = 0;

  let normalizerLoadPromise = null;

  function getInput() {
    return document.getElementById("patientInputText");
  }

  function getButton() {
    return document.getElementById("btnVoiceInput");
  }

  function getStatus() {
    return document.getElementById("voiceInputStatus");
  }

  function cleanText(value) {
    return String(value || "")
      .replace(/\s+/g, " ")
      .replace(/\s+([,.!?])/g, "$1")
      .trim();
  }

  function joinText(...values) {
    return values
      .map(cleanText)
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function ensureMedicalNormalizerLoaded() {
    if (
      window.SAVERMedicalTextNormalizer &&
      typeof window.SAVERMedicalTextNormalizer.normalize === "function"
    ) {
      return Promise.resolve(true);
    }

    if (normalizerLoadPromise) {
      return normalizerLoadPromise;
    }

    normalizerLoadPromise = new Promise((resolve) => {
      const existing = document.querySelector(
        'script[data-saver-medical-normalizer="true"]'
      );

      if (existing) {
        if (
          window.SAVERMedicalTextNormalizer &&
          typeof window.SAVERMedicalTextNormalizer.normalize === "function"
        ) {
          resolve(true);
          return;
        }

        existing.addEventListener("load", () => resolve(true), { once: true });
        existing.addEventListener("error", () => resolve(false), { once: true });
        return;
      }

      const script = document.createElement("script");
      script.src = "/js/medicalTextNormalizer.js";
      script.async = true;
      script.dataset.saverMedicalNormalizer = "true";

      script.onload = () => {
        resolve(
          Boolean(
            window.SAVERMedicalTextNormalizer &&
            typeof window.SAVERMedicalTextNormalizer.normalize === "function"
          )
        );
      };

      script.onerror = () => resolve(false);
      document.head.appendChild(script);
    });

    return normalizerLoadPromise;
  }

  function normalizeTranscript(rawText) {
    const raw = cleanText(rawText);

    if (
      window.SAVERMedicalTextNormalizer &&
      typeof window.SAVERMedicalTextNormalizer.normalize === "function"
    ) {
      return window.SAVERMedicalTextNormalizer.normalize(raw);
    }

    return {
      rawText: raw,
      normalizedText: raw,
      corrections: [],
      correctionCount: 0,
      warnings: [],
      warningCount: 0,
    };
  }

  function saveTranscriptState(result) {
    window.SAVER_VOICE_TRANSCRIPT = {
      rawText: result.rawText || "",
      normalizedText: result.normalizedText || "",
      corrections: result.corrections || [],
      correctionCount: result.correctionCount || 0,
      warnings: result.warnings || [],
      warningCount: result.warningCount || 0,
      provider: "Google Cloud Speech-to-Text V2 Streaming",
      updatedAt: new Date().toISOString(),
    };
  }

  function updateTextarea() {
    const input = getInput();
    if (!input) return;

    const rawVoiceText = joinText(finalTranscript, interimTranscript);
    const normalized = normalizeTranscript(rawVoiceText);

    saveTranscriptState(normalized);

    input.value = joinText(baseText, normalized.normalizedText);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function commitInterimIfNeeded() {
    const interim = cleanText(interimTranscript);
    if (!interim) return;

    finalTranscript = joinText(finalTranscript, interim);
    interimTranscript = "";
    updateTextarea();
  }

  function setStatus(text, type = "ready") {
    const status = getStatus();
    if (!status) return;

    status.textContent = text;

    const common =
      "text-[11px] px-2.5 py-1.5 rounded-lg border whitespace-nowrap";

    if (type === "listening") {
      status.className =
        `${common} bg-rose-500/20 text-rose-300 border-rose-500/30`;
    } else if (type === "success") {
      status.className =
        `${common} bg-emerald-500/20 text-emerald-300 border-emerald-500/30`;
    } else if (type === "error") {
      status.className =
        `${common} bg-amber-500/20 text-amber-300 border-amber-500/30`;
    } else {
      status.className =
        `${common} bg-slate-800 text-slate-400 border-slate-700`;
    }
  }

  function showStartButton() {
    const button = getButton();
    if (!button) return;

    button.disabled = false;
    button.className =
      "bg-sky-600 hover:bg-sky-500 text-white font-medium " +
      "py-2.5 px-4 rounded-lg text-xs sm:text-sm " +
      "shadow-lg shadow-sky-600/20 inline-flex items-center " +
      "justify-center gap-2 transition active:scale-95 whitespace-nowrap";

    button.innerHTML =
      '<i class="fa-solid fa-microphone"></i>' +
      '<span>음성 입력 시작</span>';
  }

  function showRecordingButton() {
    const button = getButton();
    if (!button) return;

    button.disabled = false;
    button.className =
      "bg-rose-600 hover:bg-rose-500 text-white font-medium " +
      "py-2.5 px-4 rounded-lg text-xs sm:text-sm " +
      "shadow-lg shadow-rose-600/20 inline-flex items-center " +
      "justify-center gap-2 transition active:scale-95 whitespace-nowrap";

    button.innerHTML =
      '<i class="fa-solid fa-stop"></i>' +
      '<span>음성 입력 중지</span>';
  }

  function showWaitingButton(label = "연결 중") {
    const button = getButton();
    if (!button) return;

    button.disabled = true;
    button.className =
      "bg-slate-700 text-slate-300 font-medium " +
      "py-2.5 px-4 rounded-lg text-xs sm:text-sm " +
      "inline-flex items-center justify-center gap-2 whitespace-nowrap " +
      "opacity-80 cursor-not-allowed";

    button.innerHTML =
      '<i class="fa-solid fa-spinner fa-spin"></i>' +
      `<span>${label}</span>`;
  }

  function showCompletionStatus() {
    const state = window.SAVER_VOICE_TRANSCRIPT || {};
    const corrections = Number(state.correctionCount || 0);
    const warnings = Number(state.warningCount || 0);

    if (warnings > 0) {
      setStatus(`음성 입력 완료 · 확인 필요 ${warnings}건`, "error");
      console.warn("[SAVER Voice] warnings:", state.warnings);
      return;
    }

    if (corrections > 0) {
      setStatus(`음성 입력 완료 · 후처리 ${corrections}건`, "success");
      return;
    }

    setStatus("Google STT 음성 입력 완료", "success");
  }

  function floatToPcm16Buffer(samples) {
    const buffer = new ArrayBuffer(samples.length * 2);
    const view = new DataView(buffer);

    for (let i = 0; i < samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      const value = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

      view.setInt16(i * 2, value, true);
    }

    return buffer;
  }

  function sendPcmSamples(samples) {
    if (!webSocket || webSocket.readyState !== WebSocket.OPEN) return;
    if (!samples || samples.length === 0) return;

    webSocket.send(floatToPcm16Buffer(samples));
  }

  function pushAudioSamples(samples) {
    if (!isRecording || !samples || samples.length === 0) return;

    let srcOffset = 0;

    while (srcOffset < samples.length) {
      const remaining = PCM_CHUNK_SAMPLES - pcmBufferLength;
      const copyCount = Math.min(
        remaining,
        samples.length - srcOffset
      );

      pcmBuffer.set(
        samples.subarray(
          srcOffset,
          srcOffset + copyCount
        ),
        pcmBufferLength
      );

      pcmBufferLength += copyCount;
      srcOffset += copyCount;

      if (pcmBufferLength === PCM_CHUNK_SAMPLES) {
        sendPcmSamples(pcmBuffer);

        pcmBuffer = new Float32Array(
          PCM_CHUNK_SAMPLES
        );

        pcmBufferLength = 0;
      }
    }
  }

  function flushAudioSamples() {
    if (pcmBufferLength <= 0) return;

    sendPcmSamples(
      pcmBuffer.subarray(
        0,
        pcmBufferLength
      )
    );

    pcmBuffer = new Float32Array(
      PCM_CHUNK_SAMPLES
    );

    pcmBufferLength = 0;
  }

  function stopMediaTracks() {
    if (!mediaStream) return;

    mediaStream.getTracks().forEach(
      (track) => {
        try {
          track.stop();
        } catch (_) {
          // ignore
        }
      }
    );

    mediaStream = null;
  }

  async function disconnectAudioGraph() {
    if (captureNode) {
      try {
        if (
          "port" in captureNode &&
          captureNode.port
        ) {
          captureNode.port.onmessage =
            null;
        }

        if (
          "onaudioprocess" in
          captureNode
        ) {
          captureNode.onaudioprocess =
            null;
        }

        captureNode.disconnect();
      } catch (_) {
        // ignore
      }
    }

    if (sourceNode) {
      try {
        sourceNode.disconnect();
      } catch (_) {
        // ignore
      }
    }

    if (silentGainNode) {
      try {
        silentGainNode.disconnect();
      } catch (_) {
        // ignore
      }
    }

    captureNode = null;
    sourceNode = null;
    silentGainNode = null;

    if (audioContext) {
      try {
        await audioContext.close();
      } catch (_) {
        // ignore
      }

      audioContext = null;
    }

    if (workletBlobUrl) {
      URL.revokeObjectURL(
        workletBlobUrl
      );

      workletBlobUrl = null;
    }

    stopMediaTracks();
  }

  async function createAudioCaptureGraph() {
    const AudioContextClass =
      window.AudioContext ||
      window.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error(
        "이 브라우저는 Web Audio API를 지원하지 않습니다."
      );
    }

    if (!audioContext) {
      audioContext =
        new AudioContextClass();

      await audioContext.resume();
    }

    sourceNode =
      audioContext.createMediaStreamSource(
        mediaStream
      );

    silentGainNode =
      audioContext.createGain();

    silentGainNode.gain.value = 0;

    silentGainNode.connect(
      audioContext.destination
    );

    if (
      audioContext.audioWorklet &&
      window.AudioWorkletNode
    ) {
      const workletCode = `
        class SaverPcmProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];

            if (input && input[0]) {
              this.port.postMessage(
                input[0].slice(0)
              );
            }

            return true;
          }
        }

        registerProcessor(
          'saver-pcm-processor',
          SaverPcmProcessor
        );
      `;

      const blob =
        new Blob(
          [workletCode],
          {
            type:
              "application/javascript",
          }
        );

      workletBlobUrl =
        URL.createObjectURL(blob);

      await audioContext
        .audioWorklet
        .addModule(
          workletBlobUrl
        );

      captureNode =
        new AudioWorkletNode(
          audioContext,
          "saver-pcm-processor"
        );

      captureNode.port.onmessage =
        (event) => {
          if (event.data) {
            pushAudioSamples(
              event.data
            );
          }
        };

      sourceNode.connect(
        captureNode
      );

      captureNode.connect(
        silentGainNode
      );

      return;
    }

    const processor =
      audioContext
        .createScriptProcessor(
          4096,
          1,
          1
        );

    processor.onaudioprocess =
      (event) => {
        const input =
          event.inputBuffer
            .getChannelData(0);

        pushAudioSamples(
          input
        );
      };

    captureNode = processor;

    sourceNode.connect(
      captureNode
    );

    captureNode.connect(
      silentGainNode
    );
  }

  function buildWebSocketUrl() {
    const protocol =
      window.location.protocol ===
      "https:"
        ? "wss:"
        : "ws:";

    return (
      `${protocol}//` +
      `${window.location.host}` +
      `/ws/transcribe`
    );
  }

  function closeWebSocketSilently() {
    const socket = webSocket;
    webSocket = null;

    if (!socket) return;

    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;

    try {
      if (
        socket.readyState ===
          WebSocket.OPEN ||
        socket.readyState ===
          WebSocket.CONNECTING
      ) {
        socket.close(
          1000,
          "client cleanup"
        );
      }
    } catch (_) {
      // ignore
    }
  }

  function clearTimers() {
    if (autoStopTimer) {
      clearTimeout(
        autoStopTimer
      );

      autoStopTimer = null;
    }

    if (stopGraceTimer) {
      clearTimeout(
        stopGraceTimer
      );

      stopGraceTimer = null;
    }
  }

  async function finishVoiceInputGracefully() {
    if (serverFinished) return;

    serverFinished = true;

    clearTimers();

    isConnecting = false;
    isRecording = false;
    isStopping = false;

    commitInterimIfNeeded();

    await disconnectAudioGraph();

    closeWebSocketSilently();

    showStartButton();

    showCompletionStatus();
  }

  async function handleStreamingMessage(
    message
  ) {
    if (
      !message ||
      !message.type
    ) {
      return;
    }

    if (
      message.type ===
      "ready"
    ) {
      if (
        isStopping ||
        serverFinished
      ) {
        return;
      }

      isConnecting = false;
      isRecording = true;

      showRecordingButton();

      setStatus(
        `● 지금 말씀하세요 · ${
          message.model ||
          "실시간 STT"
        }`,
        "listening"
      );

      await createAudioCaptureGraph();

      clearTimeout(
        autoStopTimer
      );

      autoStopTimer =
        setTimeout(() => {
          if (
            isRecording
          ) {
            stopVoiceInput();
          }
        }, MAX_RECORDING_MS);

      return;
    }

    if (
      message.type ===
      "transcript"
    ) {
      const text =
        cleanText(
          message.text
        );

      if (!text) {
        return;
      }

      if (
        message.isFinal
      ) {
        finalTranscript =
          joinText(
            finalTranscript,
            text
          );

        interimTranscript =
          "";
      } else {
        interimTranscript =
          text;
      }

      updateTextarea();

      if (isRecording) {
        setStatus(
          message.isFinal
            ? "계속 듣는 중 · 실시간 STT"
            : "음성 인식 중 · 실시간 STT",
          "listening"
        );
      }

      return;
    }

    if (
      message.type ===
      "done"
    ) {
      await finishVoiceInputGracefully();
      return;
    }

    if (
      message.type ===
      "error"
    ) {
      throw new Error(
        message.message ||
        "실시간 STT 오류"
      );
    }
  }

  async function startVoiceInput() {
    if (
      isConnecting ||
      isRecording ||
      isStopping
    ) {
      return;
    }

    const input = getInput();

    if (!input) {
      alert(
        "환자 상태 입력창을 찾지 못했습니다."
      );

      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices
        .getUserMedia
    ) {
      setStatus(
        "마이크 녹음 미지원",
        "error"
      );

      alert(
        "Chrome에서 실행해 주세요."
      );

      return;
    }

    if (!window.WebSocket) {
      setStatus(
        "실시간 STT 미지원",
        "error"
      );

      alert(
        "현재 브라우저에서는 WebSocket을 사용할 수 없습니다."
      );

      return;
    }

    await ensureMedicalNormalizerLoaded();

    baseText =
      cleanText(
        input.value
      );

    finalTranscript = "";
    interimTranscript = "";

    pcmBuffer =
      new Float32Array(
        PCM_CHUNK_SAMPLES
      );

    pcmBufferLength = 0;

    isConnecting = true;
    isRecording = false;
    isStopping = false;
    serverFinished = false;
    manualStopRequested = false;
    fatalHandled = false;

    window.SAVER_VOICE_TRANSCRIPT = {
      rawText: "",
      normalizedText: "",
      corrections: [],
      correctionCount: 0,
      warnings: [],
      warningCount: 0,
      provider:
        "Google Cloud Speech-to-Text V2 Streaming",
      updatedAt:
        new Date().toISOString(),
    };

    showWaitingButton(
      "실시간 STT 연결 중"
    );

    setStatus(
      "마이크 권한 확인 중",
      "ready"
    );

    try {
      mediaStream =
        await navigator
          .mediaDevices
          .getUserMedia({
            audio: {
              echoCancellation:
                true,
              noiseSuppression:
                true,
              autoGainControl:
                true,
              channelCount: 1,
            },
          });

      const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;

      audioContext =
        new AudioContextClass();

      await audioContext.resume();

      const sampleRate =
        audioContext.sampleRate;

      const socket =
        new WebSocket(
          buildWebSocketUrl()
        );

      socket.binaryType =
        "arraybuffer";

      webSocket = socket;

      socket.onopen = () => {
        if (
          socket !== webSocket ||
          serverFinished
        ) {
          return;
        }

        socket.send(
          JSON.stringify({
            type: "start",
            sampleRate,
          })
        );

        setStatus(
          "Google STT 연결 중",
          "ready"
        );
      };

      socket.onmessage =
        async (event) => {
          if (
            socket !== webSocket ||
            serverFinished
          ) {
            return;
          }

          try {
            const message =
              JSON.parse(
                event.data
              );

            await handleStreamingMessage(
              message
            );
          } catch (error) {
            console.error(
              "[SAVER Voice] streaming message error:",
              error
            );

            await failVoiceInput(
              error
            );
          }
        };

      socket.onerror =
        async () => {
          if (
            socket !== webSocket ||
            serverFinished ||
            manualStopRequested ||
            isStopping ||
            fatalHandled
          ) {
            return;
          }

          await failVoiceInput(
            new Error(
              "실시간 STT WebSocket 연결에 실패했습니다."
            )
          );
        };

      socket.onclose =
        async (event) => {
          if (
            socket !== webSocket
          ) {
            return;
          }

          if (
            serverFinished ||
            manualStopRequested ||
            isStopping
          ) {
            await finishVoiceInputGracefully();
            return;
          }

          if (!fatalHandled) {
            await failVoiceInput(
              new Error(
                `실시간 STT 연결이 예기치 않게 종료되었습니다. (code=${
                  event.code ||
                  "없음"
                })`
              ),
              true
            );
          }
        };
    } catch (error) {
      console.error(
        "[SAVER Voice] start error:",
        error
      );

      await failVoiceInput(
        error
      );
    }
  }

  async function stopVoiceInput() {
    if (
      (
        !isRecording &&
        !isConnecting
      ) ||
      isStopping
    ) {
      return;
    }

    manualStopRequested = true;
    isStopping = true;
    isRecording = false;
    isConnecting = false;

    if (autoStopTimer) {
      clearTimeout(
        autoStopTimer
      );

      autoStopTimer = null;
    }

    showWaitingButton(
      "마지막 문장 확정 중"
    );

    setStatus(
      "음성 입력 종료 중",
      "ready"
    );

    flushAudioSamples();

    await disconnectAudioGraph();

    if (
      webSocket &&
      webSocket.readyState ===
        WebSocket.OPEN
    ) {
      try {
        webSocket.send(
          JSON.stringify({
            type: "stop",
          })
        );

        stopGraceTimer =
          setTimeout(
            async () => {
              if (
                !serverFinished
              ) {
                await finishVoiceInputGracefully();
              }
            },
            STOP_GRACE_MS
          );

        return;
      } catch (error) {
        await failVoiceInput(
          error
        );

        return;
      }
    }

    await finishVoiceInputGracefully();
  }

  async function failVoiceInput(
    error,
    showAlert = true
  ) {
    if (fatalHandled) return;

    fatalHandled = true;

    console.error(
      "[SAVER Voice] error:",
      error
    );

    clearTimers();

    isConnecting = false;
    isRecording = false;
    isStopping = false;
    serverFinished = true;

    commitInterimIfNeeded();

    await disconnectAudioGraph();

    closeWebSocketSilently();

    showStartButton();

    setStatus(
      "음성 입력 오류",
      "error"
    );

    if (showAlert) {
      const message =
        error &&
        error.message
          ? error.message
          : String(error);

      alert(
        "음성 입력 처리 중 오류 발생: " +
          message
      );
    }
  }

  function toggleVoiceInput() {
    if (
      isRecording ||
      isConnecting
    ) {
      stopVoiceInput();
    } else if (
      !isStopping
    ) {
      startVoiceInput();
    }
  }

  async function initializeVoiceInput() {
    const button =
      getButton();

    if (!button) {
      console.warn(
        "[SAVER Voice] btnVoiceInput 없음"
      );

      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices
        .getUserMedia ||
      !window.WebSocket
    ) {
      button.disabled = true;

      button.classList.add(
        "opacity-50",
        "cursor-not-allowed"
      );

      setStatus(
        "실시간 음성 입력 미지원",
        "error"
      );

      return;
    }

    await ensureMedicalNormalizerLoaded();

    showStartButton();

    setStatus(
      "마이크 대기 · 실시간 Google STT",
      "ready"
    );

    console.log(
      "[SAVER Voice] Google Streaming STT ready"
    );
  }

  window.toggleVoiceInput =
    toggleVoiceInput;

  window.startVoiceInput =
    startVoiceInput;

  window.stopVoiceInput =
    stopVoiceInput;

  if (
    document.readyState ===
    "loading"
  ) {
    document.addEventListener(
      "DOMContentLoaded",
      initializeVoiceInput
    );
  } else {
    initializeVoiceInput();
  }
})();