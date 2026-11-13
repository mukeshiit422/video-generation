"""
TTS Engine – multi-backend text-to-speech with voice cloning support.

Priority chain:
  1. Coqui XTTS-v2    – best quality, multilingual, voice cloning (local)
  2. Bark (Suno)       – good quality, expressive (local, slower)
  3. Edge-TTS          – Microsoft neural TTS (free, online required)
  4. pyttsx3           – offline fallback (robotic, always works)
"""
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# transformers ≥4.46 removed BeamSearchScorer from the public namespace.
# Coqui TTS imports it as `from transformers import BeamSearchScorer` — shim it back.
try:
    import transformers as _tf
    if not hasattr(_tf, "BeamSearchScorer"):
        from transformers.generation.beam_search import BeamSearchScorer as _BSS
        _tf.BeamSearchScorer = _BSS
except Exception:
    pass

from modules.script_processor import Segment
from utils.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent

# Curated Edge-TTS neural voices offered in the UI (all verified against
# edge_tts.list_voices()). "Multilingual" voices can also read Hindi/other text.
VOICES = [
    # id                                  label                         gender    group
    ("en-IN-NeerjaNeural",            "Neerja – warm, friendly",        "female", "English (India)"),
    ("en-IN-NeerjaExpressiveNeural",  "Neerja Expressive – lively",     "female", "English (India)"),
    ("en-IN-PrabhatNeural",           "Prabhat – friendly",             "male",   "English (India)"),
    ("hi-IN-SwaraNeural",             "Swara – friendly",               "female", "Hindi"),
    ("hi-IN-MadhurNeural",            "Madhur – friendly",              "male",   "Hindi"),
    ("en-US-AvaMultilingualNeural",   "Ava – expressive, caring",       "female", "English (US)"),
    ("en-US-EmmaMultilingualNeural",  "Emma – cheerful, clear",         "female", "English (US)"),
    ("en-US-JennyNeural",             "Jenny – friendly",               "female", "English (US)"),
    ("en-US-AriaNeural",              "Aria – confident",               "female", "English (US)"),
    ("en-US-AndrewMultilingualNeural","Andrew – warm, confident",       "male",   "English (US)"),
    ("en-US-BrianMultilingualNeural", "Brian – casual, approachable",   "male",   "English (US)"),
    ("en-US-GuyNeural",               "Guy – energetic",                "male",   "English (US)"),
    ("en-US-ChristopherNeural",       "Christopher – authoritative",    "male",   "English (US)"),
    ("en-GB-SoniaNeural",             "Sonia – friendly",               "female", "English (UK)"),
    ("en-GB-LibbyNeural",             "Libby – friendly",               "female", "English (UK)"),
    ("en-GB-RyanNeural",              "Ryan – friendly",                "male",   "English (UK)"),
    ("en-AU-NatashaNeural",           "Natasha – friendly",             "female", "English (Australia)"),
    ("en-US-AnaNeural",               "Ana – child voice (cute)",       "female", "Kids"),
]
EDGE_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-\w+Neural$")


class TTSEngine:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg.get("tts", {})
        self.device = device
        self._backend = None
        self._tts_model = None
        self._backend_name = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        segments: List[Segment],
        voice_sample: Optional[str] = None,
        language: str = "en",
        speaker: str = "default",
        output_path: str = "outputs/temp_audio.wav",
        rate: str = "+0%",
    ) -> str:
        """Generate a single audio file from all segments.

        speaker: an Edge-TTS voice id (see VOICES) to use that voice directly, or
                 "default" for the configured backend chain. A voice_sample
                 (cloning) always takes the backend chain, since only XTTS clones.
        rate:    speaking speed for Edge-TTS, e.g. "-10%", "+0%", "+15%".
        Falls back to silence if every TTS backend fails."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._rate = rate

        if EDGE_VOICE_RE.match(speaker or "") and not voice_sample:
            backend, self._backend_name = self._synth_edge_tts, f"edge_tts ({speaker})"
        else:
            try:
                backend = self._load_backend(voice_sample)
            except RuntimeError as e:
                logger.warning(f"All TTS backends failed ({e}). Generating silence.")
                return self._generate_silence(segments, output_path)

        logger.info(f"TTS backend: {self._backend_name}")
        combined = AudioSegment.silent(duration=0)
        self.failed_segments = 0  # callers can tell "spoke" from "silent fallback"

        for seg in segments:
            logger.debug(f"  Synthesising segment {seg.index}: '{seg.text[:60]}…'")
            try:
                wav_path = backend(seg.text, voice_sample, language, speaker)
                audio = AudioSegment.from_wav(wav_path)
            except Exception as e:
                logger.warning(f"  Segment {seg.index} failed ({e}), using silence.")
                self.failed_segments += 1
                # Estimate duration: ~130 words/min → chars/650 seconds
                dur_ms = max(1000, int(len(seg.text) / 650 * 60 * 1000))
                audio = AudioSegment.silent(duration=dur_ms)

            combined += audio
            if seg.pause_after > 0:
                combined += AudioSegment.silent(duration=int(seg.pause_after * 1000))

        # Guard: SadTalker errors on a 0-byte or sub-100ms file
        if len(combined) < 500:
            combined += AudioSegment.silent(duration=1000)

        combined.export(output_path, format="wav")
        logger.info(f"Audio saved → {output_path} ({len(combined)/1000:.1f}s)")
        return output_path

    def _generate_silence(self, segments: List[Segment], output_path: str) -> str:
        """Estimate speech duration from text length and write a silent WAV."""
        total_chars = sum(len(s.text) for s in segments)
        total_pauses = sum(s.pause_after for s in segments)
        # ~15 chars/second average speech rate
        duration_ms = max(2000, int(total_chars / 15 * 1000) + int(total_pauses * 1000))
        silence = AudioSegment.silent(duration=duration_ms)
        silence.export(output_path, format="wav")
        logger.info(f"Silence audio ({duration_ms/1000:.1f}s) saved → {output_path}")
        return output_path

    def synthesize_text(
        self,
        text: str,
        voice_sample: Optional[str] = None,
        language: str = "en",
        output_path: str = "outputs/temp_audio.wav",
        speaker: str = "default",
        rate: str = "+0%",
    ) -> str:
        """Convenience: synthesise raw text string directly."""
        from modules.script_processor import ScriptProcessor, Segment
        segments = [Segment(index=0, text=text)]
        return self.synthesize(segments, voice_sample, language, speaker, output_path, rate)

    # ── Backend Loader ─────────────────────────────────────────────────────────

    def _load_backend(self, voice_sample: Optional[str]):
        if self._backend is not None:
            return self._backend

        priority = [self.cfg.get("primary_backend", "xtts_v2")] + \
                   self.cfg.get("fallback_backends", ["edge_tts", "pyttsx3"])

        for name in priority:
            fn = getattr(self, f"_backend_{name}", None)
            if fn is None:
                continue
            try:
                probe = fn  # test init below
                if name == "xtts_v2":
                    self._init_xtts()
                elif name == "bark":
                    self._init_bark()
                self._backend = getattr(self, f"_synth_{name}")
                self._backend_name = name
                logger.info(f"TTS backend initialised: {name}")
                return self._backend
            except Exception as e:
                logger.warning(f"Backend '{name}' unavailable: {e}")

        raise RuntimeError("All TTS backends failed. Install TTS, bark, edge-tts, or pyttsx3.")

    # ── XTTS-v2 (Coqui) ───────────────────────────────────────────────────────

    def _init_xtts(self):
        # Suppress the interactive CPML license prompt; we agree to non-commercial use
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # PyTorch 2.6+ sets weights_only=True by default.
        # XTTS checkpoints reference many internal Coqui classes; patching torch.load
        # for this trusted local model is simpler than allowlisting every class.
        import torch
        import functools
        _orig_load = torch.load
        @functools.wraps(_orig_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load


        from TTS.api import TTS
        model_name = self.cfg.get("xtts_v2", {}).get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        logger.info(f"Loading XTTS-v2 model: {model_name}")
        self._tts_model = TTS(model_name).to(self.device)
        logger.info("XTTS-v2 loaded")

    def _synth_xtts_v2(
        self,
        text: str,
        voice_sample: Optional[str],
        language: str,
        speaker: str,
    ) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name

        kwargs = dict(
            text=text,
            language=language,
            file_path=tmp,
            speed=self.cfg.get("xtts_v2", {}).get("speed", 1.0),
        )
        if voice_sample and Path(voice_sample).exists():
            kwargs["speaker_wav"] = voice_sample
        else:
            kwargs["speaker"] = speaker if speaker != "default" else "Ana Florence"

        self._tts_model.tts_to_file(**kwargs)
        return tmp

    def _backend_xtts_v2(self):
        pass  # probe sentinel

    # ── Bark (Suno) ───────────────────────────────────────────────────────────

    def _init_bark(self):
        import bark
        from bark import preload_models
        logger.info("Loading Bark models (first run downloads ~2 GB)...")
        preload_models()
        self._bark = bark

    def _synth_bark(
        self,
        text: str,
        voice_sample: Optional[str],
        language: str,
        speaker: str,
    ) -> str:
        from bark import generate_audio, SAMPLE_RATE
        preset = self.cfg.get("bark", {}).get("voice_preset", "v2/en_speaker_6")
        audio_array = generate_audio(text, history_prompt=preset)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, audio_array, SAMPLE_RATE)
        return tmp

    def _backend_bark(self):
        pass

    # ── Edge-TTS (Microsoft, free, online) ───────────────────────────────────

    def _synth_edge_tts(
        self,
        text: str,
        voice_sample: Optional[str],
        language: str,
        speaker: str,
    ) -> str:
        import asyncio
        import edge_tts

        voice = self.cfg.get("edge_tts", {}).get("voice", "en-US-JennyNeural")
        # Pick voice by language
        lang_voices = {
            "hi": "hi-IN-SwaraNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
        }
        if language in lang_voices:
            voice = lang_voices[language]
        if EDGE_VOICE_RE.match(speaker or ""):
            voice = speaker  # explicit choice from the UI wins
        rate = getattr(self, "_rate", None) or self.cfg.get("edge_tts", {}).get("rate", "+0%")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name

        async def _run():
            tts = edge_tts.Communicate(text, voice, rate=rate)
            await tts.save(tmp_mp3)

        asyncio.run(_run())

        # Convert mp3 → wav
        tmp_wav = tmp_mp3.replace(".mp3", ".wav")
        AudioSegment.from_mp3(tmp_mp3).export(tmp_wav, format="wav")
        os.unlink(tmp_mp3)
        return tmp_wav

    def _backend_edge_tts(self):
        import edge_tts

    # ── pyttsx3 (offline fallback) ────────────────────────────────────────────

    def _synth_pyttsx3(
        self,
        text: str,
        voice_sample: Optional[str],
        language: str,
        speaker: str,
    ) -> str:
        import pyttsx3

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        return tmp

    def _backend_pyttsx3(self):
        import pyttsx3
