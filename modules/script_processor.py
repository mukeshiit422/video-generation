"""
Script Processor – splits a raw script into timed segments suitable for TTS.

Supports scene markers:  [SCENE], [CUT], [PAUSE], [BREAK], [LONG_PAUSE]
Automatically splits long passages at sentence boundaries.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Segment:
    index: int
    text: str
    pause_after: float = 0.0        # seconds of silence after this segment
    emotion: str = "neutral"        # hint for TTS prosody
    is_scene_break: bool = False


class ScriptProcessor:
    SCENE_MARKERS = {
        r"\[SCENE\]":      0.5,
        r"\[CUT\]":        0.3,
        r"\[PAUSE\]":      0.5,
        r"\[BREAK\]":      1.0,
        r"\[LONG_PAUSE\]": 2.0,
        r"\[PAUSE:(\d+(?:\.\d+)?)\]": None,   # [PAUSE:1.5]
    }

    EMOTION_PATTERNS = {
        "happy":    r"\[HAPPY\]|\[EXCITED\]",
        "sad":      r"\[SAD\]|\[MELANCHOLY\]",
        "angry":    r"\[ANGRY\]|\[INTENSE\]",
        "whisper":  r"\[WHISPER\]|\[SOFT\]",
        "energetic":r"\[ENERGETIC\]|\[LOUD\]",
    }

    def __init__(self, cfg: dict):
        self.max_chars = cfg.get("script_processing", {}).get("max_segment_chars", 300)

    def process(self, script: str) -> List[Segment]:
        """Parse a full script into ordered Segment objects."""
        logger.debug(f"Processing script ({len(script)} chars)...")

        # Normalize line endings
        text = script.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3+ blank lines → double blank
        text = re.sub(r"\n{3,}", "\n\n", text)

        raw_segments = self._split_on_markers(text)
        segments: List[Segment] = []

        for idx, (raw_text, pause) in enumerate(raw_segments):
            emotion = self._extract_emotion(raw_text)
            clean = self._clean_text(raw_text)
            if not clean.strip():
                continue

            # Further split if too long
            sub_texts = self._split_long_text(clean)
            for i, sub in enumerate(sub_texts):
                is_last = i == len(sub_texts) - 1
                segments.append(Segment(
                    index=len(segments),
                    text=sub,
                    pause_after=pause if is_last else 0.1,
                    emotion=emotion,
                    is_scene_break=(pause >= 0.5 and is_last),
                ))

        logger.info(f"Script split into {len(segments)} segments")
        return segments

    def _split_on_markers(self, text: str):
        """Split text on scene/pause markers, return (text, pause_seconds) pairs."""
        result = []
        # Build combined marker pattern
        marker_pattern = "|".join(self.SCENE_MARKERS.keys())
        parts = re.split(f"({marker_pattern})", text, flags=re.IGNORECASE)

        current_text = ""
        current_pause = 0.0

        for part in parts:
            matched_pause = self._match_marker(part)
            if matched_pause is not None:
                if current_text.strip():
                    result.append((current_text.strip(), max(current_pause, matched_pause)))
                    current_text = ""
                current_pause = matched_pause
            else:
                current_text += part or ""
                if current_pause > 0:
                    current_pause = 0.0

        if current_text.strip():
            result.append((current_text.strip(), current_pause))

        return result if result else [(text.strip(), 0.0)]

    def _match_marker(self, s) -> Optional[float]:
        """Return pause duration if s is a marker, else None."""
        if s is None:
            return None
        s = s.strip()
        for pattern, pause in self.SCENE_MARKERS.items():
            m = re.fullmatch(pattern, s, re.IGNORECASE)
            if m:
                if pause is None:  # dynamic [PAUSE:X]
                    return float(m.group(1))
                return pause
        return None

    def _extract_emotion(self, text: str) -> str:
        for emotion, pattern in self.EMOTION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return emotion
        return "neutral"

    def _clean_text(self, text: str) -> str:
        # Remove all marker tags
        clean = re.sub(r"\[(?:SCENE|CUT|PAUSE|BREAK|LONG_PAUSE|HAPPY|SAD|ANGRY|WHISPER|SOFT|EXCITED|MELANCHOLY|INTENSE|ENERGETIC|LOUD)(?::\d+(?:\.\d+)?)?\]", "", text, flags=re.IGNORECASE)
        # Collapse extra whitespace
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n+", " ", clean)
        return clean.strip()

    def _split_long_text(self, text: str) -> List[str]:
        """Split text that exceeds max_chars at sentence boundaries."""
        if len(text) <= self.max_chars:
            return [text]

        # Sentence boundary split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.max_chars:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                # Handle single sentence longer than max
                if len(sent) > self.max_chars:
                    chunks.extend(self._split_at_commas(sent))
                    current = ""
                else:
                    current = sent

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:self.max_chars]]

    def _split_at_commas(self, text: str) -> List[str]:
        """Last resort: split at commas."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        chunks, current = [], ""
        for p in parts:
            if len(current) + len(p) + 2 <= self.max_chars:
                current = (current + ", " + p).strip(", ")
            else:
                if current:
                    chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:self.max_chars]]
