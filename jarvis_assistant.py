#!/usr/bin/env python3
"""
Jarvis voice assistant — desktop GUI (PySide6).

Speak naturally; Jarvis auto-detects when you start/stop talking (VAD),
transcribes you with ElevenLabs Speech-to-Text, sends the conversation to
Claude (with tool use for actions like opening a song/site/app), and speaks
the reply back with ElevenLabs Text-to-Speech.

Run:
  python -m pip install -r requirements.txt
  python jarvis_assistant.py

Required in .env (next to this script):
  ANTHROPIC_API_KEY
  ELEVENLABS_API_KEY
  ELEVENLABS_VOICE_ID
Optional:
  ANTHROPIC_MODEL          (default: claude-sonnet-4-6)
  ELEVENLABS_STT_MODEL_ID  (default: scribe_v1)
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from PySide6.QtCore import QEasingCurve, QPointF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jarvis")

SAMPLE_RATE = 16000
BLOCK_MS = 30
CHANNELS = 1

VAD_SPIKE_RATIO = 4.5
VAD_MIN_RMS = 0.01
VAD_SILENCE_S = 0.9
VAD_MIN_SPEECH_S = 0.3
VAD_MAX_SPEECH_S = 30.0
NOISE_FLOOR_ALPHA = 0.98

ANTHROPIC_MODEL = (os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip()
ELEVENLABS_STT_MODEL_ID = (os.environ.get("ELEVENLABS_STT_MODEL_ID") or "scribe_v1").strip()
ELEVENLABS_STT_LANGUAGE = (os.environ.get("ELEVENLABS_STT_LANGUAGE") or "it").strip()
ELEVENLABS_REQUEST_TIMEOUT_S = 20.0

STARTUP_SONG_DIR = Path(__file__).resolve().parent / "song"
MIN_LOADING_SONG_S = 20.0
WELCOME_PHRASE = "Buongiorno soldato Piscioneri, come posso servirti oggi?"

PIPER_VOICES_DIR = Path(__file__).resolve().parent / ".piper_voices"
PIPER_VOICE_MODEL = (os.environ.get("PIPER_VOICE_MODEL") or "it_IT-paola-medium").strip()

SYSTEM_PROMPT = (
    "You are Jarvis, a concise, friendly voice assistant running on the user's Mac. "
    "Always reply in Italian unless the user clearly speaks another language. Keep "
    "replies short and natural, suitable for being read aloud. Never use emojis or "
    "emoticons in your replies. Use the available tools when the user asks you to "
    "open a website, play music, or launch an application; otherwise just answer "
    "directly."
)

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text).strip()

TOOLS = [
    {
        "name": "open_url",
        "description": "Open a URL in the default web browser (or Chrome if available).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to open."}},
            "required": ["url"],
        },
    },
    {
        "name": "open_app",
        "description": "Launch a macOS application by name (e.g. 'Spotify', 'Cursor', 'Notes').",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Application name."}},
            "required": ["name"],
        },
    },
    {
        "name": "play_spotify",
        "description": "Open a Spotify track/album/playlist by URI or URL, or a search query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uri_or_query": {
                    "type": "string",
                    "description": "Spotify URI/URL, or free-text search query.",
                }
            },
            "required": ["uri_or_query"],
        },
    },
]


def tool_open_url(url: str) -> str:
    url = url.strip()
    if not url:
        return "No URL given."
    if sys.platform == "darwin" and shutil.which("open"):
        chrome = "/Applications/Google Chrome.app"
        if Path(chrome).exists():
            subprocess.Popen(["open", "-a", "Google Chrome", url])
        else:
            subprocess.Popen(["open", url])
    else:
        webbrowser.open(url)
    return f"Opened {url}"


def tool_open_app(name: str) -> str:
    name = name.strip()
    if not name:
        return "No app name given."
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", "-a", name])
            return f"Launched {name}"
        except OSError as e:
            return f"Could not launch {name}: {e}"
    exe = shutil.which(name.lower())
    if exe:
        subprocess.Popen([exe])
        return f"Launched {name}"
    return f"Could not find {name} on this system."


def tool_play_spotify(uri_or_query: str) -> str:
    q = uri_or_query.strip()
    if not q:
        return "No query given."
    if q.startswith("spotify:") or q.startswith("http"):
        url = q
    else:
        url = f"https://open.spotify.com/search/{q}"
    if sys.platform == "darwin":
        subprocess.Popen(["open", url])
    else:
        webbrowser.open(url)
    return f"Opened in Spotify: {q}"


TOOL_IMPLS = {
    "open_url": lambda inp: tool_open_url(inp.get("url", "")),
    "open_app": lambda inp: tool_open_app(inp.get("name", "")),
    "play_spotify": lambda inp: tool_play_spotify(inp.get("uri_or_query", "")),
}


def rms_mono(block: np.ndarray) -> float:
    if block.ndim > 1:
        block = np.mean(block.astype(np.float64), axis=1)
    else:
        block = block.astype(np.float64)
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))


def pcm_f32_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    pcm_i16 = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def speak(client, text: str) -> str | None:
    """Returns an error message on failure, or None on success.

    Uses Piper (free, offline, local neural TTS) instead of a paid TTS API.
    """
    text = strip_emojis(text)
    if not text:
        return None
    model_path = PIPER_VOICES_DIR / f"{PIPER_VOICE_MODEL}.onnx"
    if not model_path.is_file():
        return f"Piper voice model not found: {model_path}"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        subprocess.run(
            ["piper", "-m", str(model_path), "-f", str(wav_path)],
            input=text,
            text=True,
            check=True,
            capture_output=True,
        )
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm_i16 = np.frombuffer(raw, dtype=np.int16)
        pcm_f = pcm_i16.astype(np.float32) / 32768.0
        sd.play(pcm_f, rate)
        sd.wait()
    except (OSError, subprocess.CalledProcessError, wave.Error) as e:
        return f"TTS failed: {e}"
    finally:
        wav_path.unlink(missing_ok=True)
    return None


def transcribe(client, pcm: np.ndarray, sample_rate: int) -> str:
    from elevenlabs.core.request_options import RequestOptions

    wav_bytes = pcm_f32_to_wav_bytes(pcm, sample_rate)
    result = client.speech_to_text.convert(
        model_id=ELEVENLABS_STT_MODEL_ID,
        file=("speech.wav", io.BytesIO(wav_bytes), "audio/wav"),
        language_code=ELEVENLABS_STT_LANGUAGE or None,
        request_options=RequestOptions(timeout_in_seconds=ELEVENLABS_REQUEST_TIMEOUT_S),
    )
    return (getattr(result, "text", "") or "").strip()


def ask_claude(anthropic_client, history: list[dict]) -> str:
    """Runs the tool-use loop against Claude and returns the final text reply."""
    messages = list(history)
    final_text = ""
    for _ in range(5):
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        final_text = "\n".join(text_parts).strip()

        if response.stop_reason != "tool_use":
            history.append({"role": "assistant", "content": response.content})
            break

        history.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            impl = TOOL_IMPLS.get(block.name)
            result = impl(block.input) if impl else f"Unknown tool: {block.name}"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        history.append({"role": "user", "content": tool_results})
        messages = history
    return final_text


class ListenerThread(QThread):
    speech_captured = Signal(np.ndarray)
    status_changed = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop = False
        self._paused = False

    def stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def run(self) -> None:
        blocksize = max(int(SAMPLE_RATE * BLOCK_MS / 1000), 1)
        noise_floor = 1e-4
        buffer: list[np.ndarray] = []
        speaking = False
        silence_blocks = 0
        silence_needed = max(1, int(VAD_SILENCE_S * 1000 / BLOCK_MS))
        max_blocks = max(1, int(VAD_MAX_SPEECH_S * 1000 / BLOCK_MS))

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=blocksize,
            ) as stream:
                self.status_changed.emit("listening")
                while not self._stop:
                    data, overflowed = stream.read(blocksize)

                    if self._paused:
                        speaking = False
                        buffer = []
                        continue

                    level = rms_mono(data)

                    if not speaking:
                        noise_floor = NOISE_FLOOR_ALPHA * noise_floor + (
                            1.0 - NOISE_FLOOR_ALPHA
                        ) * level
                        noise_floor = max(noise_floor, 1e-6)

                    threshold = max(noise_floor * VAD_SPIKE_RATIO, VAD_MIN_RMS)

                    if not speaking:
                        if level >= threshold:
                            speaking = True
                            silence_blocks = 0
                            buffer = [data.copy()]
                            self.status_changed.emit("speaking_detected")
                    else:
                        buffer.append(data.copy())
                        if level < threshold * 0.6:
                            silence_blocks += 1
                        else:
                            silence_blocks = 0

                        too_long = len(buffer) >= max_blocks
                        if silence_blocks >= silence_needed or too_long:
                            speaking = False
                            segment = np.concatenate(buffer, axis=0).flatten()
                            duration_s = len(segment) / SAMPLE_RATE
                            buffer = []
                            if duration_s >= VAD_MIN_SPEECH_S:
                                self.speech_captured.emit(segment)
                                self.status_changed.emit("processing")
                            else:
                                self.status_changed.emit("listening")
        except sd.PortAudioError as e:
            self.error.emit(str(e))


class ProcessingThread(QThread):
    transcribed = Signal(str)
    replied = Signal(str)
    failed = Signal(str)
    finished_processing = Signal()

    def __init__(self, elevenlabs_client, anthropic_client, history: list[dict], segment: np.ndarray) -> None:
        super().__init__()
        self.elevenlabs_client = elevenlabs_client
        self.anthropic_client = anthropic_client
        self.history = history
        self.segment = segment

    def run(self) -> None:
        try:
            text = transcribe(self.elevenlabs_client, self.segment, SAMPLE_RATE)
        except Exception as e:
            self.failed.emit(f"Transcription failed: {e}")
            self.finished_processing.emit()
            return
        if not text:
            self.finished_processing.emit()
            return

        self.transcribed.emit(text)
        self.history.append({"role": "user", "content": text})

        try:
            reply = ask_claude(self.anthropic_client, self.history)
        except Exception as e:
            self.failed.emit(f"Claude request failed: {e}")
            self.finished_processing.emit()
            return

        if reply:
            reply = strip_emojis(reply)
            self.replied.emit(reply)
            tts_error = speak(self.elevenlabs_client, reply)
            if tts_error:
                self.failed.emit(tts_error)
        self.finished_processing.emit()


class Orb(QWidget):
    """Pulsing glowing orb that reflects Jarvis's current state."""

    COLORS = {
        "listening": QColor(79, 209, 255),
        "speaking_detected": QColor(120, 255, 200),
        "processing": QColor(186, 130, 255),
        "speaking": QColor(255, 190, 110),
    }

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(220, 220)
        self.state = "listening"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_state(self, state: str) -> None:
        self.state = state if state in self.COLORS else "listening"

    def _tick(self) -> None:
        speed = {"listening": 0.04, "speaking_detected": 0.16, "processing": 0.12, "speaking": 0.14}
        self._phase += speed.get(self.state, 0.05)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = self.COLORS.get(self.state, self.COLORS["listening"])

        import math

        pulse = 0.5 + 0.5 * math.sin(self._phase)
        base_radius = 32 + pulse * 14
        center = QPointF(self.width() / 2, self.height() / 2)

        glow = QRadialGradient(center, base_radius * 2.2)
        glow_color = QColor(color)
        glow_color.setAlpha(90)
        glow.setColorAt(0.0, glow_color)
        transparent = QColor(color)
        transparent.setAlpha(0)
        glow.setColorAt(1.0, transparent)
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, base_radius * 2.2, base_radius * 2.2)

        core = QRadialGradient(center, base_radius)
        bright = QColor(color).lighter(140)
        core.setColorAt(0.0, bright)
        core.setColorAt(1.0, color)
        painter.setBrush(core)
        painter.drawEllipse(center, base_radius, base_radius)


class WelcomeThread(QThread):
    done = Signal()

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def run(self) -> None:
        speak(None, self.text)
        self.done.emit()


class JarvisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis")
        self.resize(480, 620)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._drag_pos = None

        self.setStyleSheet(
            """
            QWidget#glass {
                background-color: rgba(20, 25, 34, 215);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 26px;
            }
            QLabel#status {
                color: #cfe9ff;
                font-size: 14px;
                letter-spacing: 3px;
                font-weight: 600;
            }
            QLabel#title {
                color: #f2f8ff;
                letter-spacing: 6px;
            }
            QPushButton#closeBtn {
                background-color: rgba(255, 255, 255, 18);
                color: #d7e6f5;
                border: none;
                border-radius: 13px;
                font-size: 14px;
            }
            QPushButton#closeBtn:hover {
                background-color: rgba(255, 90, 90, 160);
            }
            QTextEdit {
                background-color: rgba(255, 255, 255, 10);
                color: #e6f0fa;
                border: 1px solid rgba(255, 255, 255, 25);
                border-radius: 18px;
                padding: 14px;
                font-size: 14px;
                selection-background-color: rgba(79, 209, 255, 120);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 60);
                border-radius: 4px;
            }
            """
        )

        central = QWidget()
        central.setObjectName("glass")
        shadow = QGraphicsDropShadowEffect(central)
        shadow.setBlurRadius(60)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 160))
        central.setGraphicsEffect(shadow)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        titlebar = QWidget()
        titlebar_layout = QHBoxLayout(titlebar)
        titlebar_layout.setContentsMargins(20, 14, 14, 0)
        close_btn = QPushButton("×")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(26, 26)
        close_btn.setToolTip("Interrompi e chiudi Jarvis")
        close_btn.clicked.connect(self.confirm_quit)
        titlebar_layout.addStretch(1)
        titlebar_layout.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.setContentsMargins(32, 4, 32, 28)
        layout.setSpacing(14)

        title = QLabel("J A R V I S")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Helvetica Neue", 26, QFont.Thin))

        self.orb = Orb()
        orb_row = QHBoxLayout()
        orb_row.addStretch(1)
        orb_row.addWidget(self.orb)
        orb_row.addStretch(1)

        self.status_label = QLabel("INITIALIZING…")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.conversation = QTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setFrameStyle(0)

        layout.addWidget(title)
        layout.addLayout(orb_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.conversation)

        outer.addWidget(titlebar)
        outer.addLayout(layout)
        self.setCentralWidget(central)

        self.elevenlabs_client = None
        self.anthropic_client = None
        self.history: list[dict] = []
        self._ready = False
        self._min_song_elapsed = True
        self._welcome_said = False
        self._start_loading_song()
        self._init_clients()

        self.listener = ListenerThread()
        self.listener.speech_captured.connect(self.on_speech_captured)
        self.listener.status_changed.connect(self.on_status_changed)
        self.listener.error.connect(self.on_error)
        self.listener.start()
        self.listener.pause()

    def _start_loading_song(self) -> None:
        tracks = []
        if STARTUP_SONG_DIR.is_dir():
            tracks = sorted(
                p for p in STARTUP_SONG_DIR.iterdir() if p.suffix.lower() in (".mp3", ".m4a", ".wav")
            )
        if not tracks:
            self.loading_player = None
            return
        self.loading_audio_output = QAudioOutput()
        self.loading_player = QMediaPlayer()
        self.loading_player.setAudioOutput(self.loading_audio_output)
        self.loading_player.setSource(QUrl.fromLocalFile(str(tracks[0])))
        self.loading_player.play()

        self._min_song_elapsed = False
        self._loading_timer = QTimer(self)
        self._loading_timer.setSingleShot(True)
        self._loading_timer.timeout.connect(self._on_min_song_elapsed)
        self._loading_timer.start(int(MIN_LOADING_SONG_S * 1000))

    def _stop_loading_song(self) -> None:
        player = getattr(self, "loading_player", None)
        if player is not None:
            player.stop()
            self.loading_player = None
        if self._welcome_said:
            self.listener.resume()
            return
        self._welcome_said = True
        self.append_line("jarvis", WELCOME_PHRASE)
        self.status_label.setText("SPEAKING…")
        self.orb.set_state("speaking")
        self.welcome_thread = WelcomeThread(WELCOME_PHRASE)
        self.welcome_thread.done.connect(self._on_welcome_done)
        self.welcome_thread.start()

    def _on_welcome_done(self) -> None:
        self.status_label.setText("LISTENING…")
        self.orb.set_state("listening")
        self.listener.resume()

    def _on_min_song_elapsed(self) -> None:
        self._min_song_elapsed = True
        if self._ready:
            self._stop_loading_song()

    def _maybe_stop_loading_song(self) -> None:
        if self._ready and self._min_song_elapsed:
            self._stop_loading_song()

    def confirm_quit(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Chiudere Jarvis?")
        box.setText("Sei sicuro di voler interrompere e chiudere Jarvis?")
        box.setIcon(QMessageBox.Question)
        cancel_btn = box.addButton("Annulla", QMessageBox.RejectRole)
        proceed_btn = box.addButton("Procedi", QMessageBox.AcceptRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is proceed_btn:
            self.close()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._drag_pos = None

    def _init_clients(self) -> None:
        api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
        anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        missing = []
        if not api_key:
            missing.append("ELEVENLABS_API_KEY")
        if not anthropic_key:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            self.append_line("system", f"Missing in .env: {', '.join(missing)}")
            return
        from elevenlabs.client import ElevenLabs
        import anthropic

        self.elevenlabs_client = ElevenLabs(api_key=api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

    def append_line(self, who: str, text: str) -> None:
        if who == "system":
            log.warning(text)
            return
        color = {"user": "#8fd6ff", "jarvis": "#4fd1ff"}.get(who, "#d7e6f5")
        label = {"user": "You", "jarvis": "Jarvis"}.get(who, who)
        self.conversation.append(
            f'<span style="color:{color}"><b>{label}:</b> {text}</span>'
        )

    def on_status_changed(self, status: str) -> None:
        if status == "listening" and not self._ready:
            self._ready = True
            self._maybe_stop_loading_song()
        labels = {
            "listening": "LISTENING…",
            "speaking_detected": "HEARING YOU…",
            "processing": "THINKING…",
        }
        self.status_label.setText(labels.get(status, status.upper()))
        self.orb.set_state(status)

    def on_error(self, message: str) -> None:
        self.append_line("system", f"Audio error: {message}")

    def on_speech_captured(self, segment: np.ndarray) -> None:
        if self.elevenlabs_client is None or self.anthropic_client is None:
            self.append_line("system", "Clients not configured; check .env.")
            return
        self.listener.pause()
        self.processing = ProcessingThread(
            self.elevenlabs_client, self.anthropic_client, self.history, segment
        )
        self.processing.transcribed.connect(lambda t: self.append_line("user", t))
        self.processing.replied.connect(self._on_replied)
        self.processing.failed.connect(lambda msg: self.append_line("system", msg))
        self.processing.finished_processing.connect(self._on_processing_done)
        self.processing.start()

    def _on_replied(self, reply: str) -> None:
        self.append_line("jarvis", reply)
        self.status_label.setText("SPEAKING…")
        self.orb.set_state("speaking")

    def _on_processing_done(self) -> None:
        self.status_label.setText("LISTENING…")
        self.orb.set_state("listening")
        self.listener.resume()

    def closeEvent(self, event) -> None:
        player = getattr(self, "loading_player", None)
        if player is not None:
            player.stop()
            self.loading_player = None
        self.listener.stop()
        if not self.listener.wait(3000):
            self.listener.terminate()
            self.listener.wait()
        for attr in ("processing", "welcome_thread"):
            thread = getattr(self, attr, None)
            if thread is not None and thread.isRunning():
                if not thread.wait(1000):
                    thread.terminate()
                    thread.wait()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = JarvisWindow()
    window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
