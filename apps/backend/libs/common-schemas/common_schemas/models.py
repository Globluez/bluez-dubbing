from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Dict, Any
from pathlib import Path
from asyncio import Lock, Future, Event
from dataclasses import dataclass, field as dataclass_field

# ASR
class ASRRequest(BaseModel):
    audio_url: str
    language_hint: str | None = None
    min_speakers: Optional[int] = None  # optional number of speakers for diarization (min, max) if known
    max_speakers: Optional[int] = None
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)

class Word(BaseModel):
    start: float
    end: float
    text: str
    score: float | None = None
    speaker_id: str | None = None

class Segment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    words: List[Word] | None = None
    speaker_id: str | None = None
    lang: str | None = None
    audio_url: str | None = None  # Optional field for audio link to that segment
    segment_id: str | None = None

# will serve for both ASR and Translation responses
class ASRResponse(BaseModel):
    segments: List[Segment] = Field(default_factory=list)
    WordSegments: List[Word] | None = None
    language: str | None = None
    audio_url: str | None = None  # Optional field for audio link to that transcription
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)


# Translate
class TranslateRequest(BaseModel):
    segments: List[Segment] | None = Field(default_factory=list)
    source_lang: str | None = None
    target_lang: str
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)


# TTS
class SegmentAudioIn(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    speaker_id: str | None = None
    lang: str | None = None
    audio_prompt_url: str | None = None
    segment_id: str | None = None
    legacy_audio_path: str | None = None

class SegmentAudioOut(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str | None = None
    audio_prompt_url: str | None = None # the audio chunk used as prompt to generate this segment (it will serve for the review stage)
    audio_url: str 
    speaker_id: str | None = None
    lang: str | None = None
    sample_rate: int | None = None
    segment_id: str | None = None

class TTSRequest(BaseModel):
    segments: List[SegmentAudioIn] = Field(default_factory=list)
    workspace: str | None = None
    language: str | None = None
    # Free-form bucket for future-proofing
    extra: Dict[str, Any] = Field(default_factory=dict)
    

class TTSResponse(BaseModel):
    segments: List[SegmentAudioOut] = Field(default_factory=list)
    meta: Dict[str, Any] | None = None

class SubtitleSegment(BaseModel):
    """Represents a subtitle segment with timing and text."""
    start: float
    end: float
    text: str
    lines: List[str]
    
    @computed_field
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @computed_field
    @property
    def cps(self) -> float:
        """Characters per second."""
        return self.char_count / self.duration if self.duration > 0 else 0

@dataclass
class TTSReviewSession():
    run_id: str
    language: str
    tts_model: str
    workspace: Path
    translation: Dict[str, Segment]
    tts_response: TTSResponse
    segments: Dict[str, SegmentAudioOut]
    lock: Lock
    future: Future[bool]
    languages: List[str]
    activate_event: Event
    segment_locks: Dict[str, Lock] = dataclass_field(default_factory=dict)

@dataclass
class TranscriptionReviewSession():
    run_id: str
    audio_duration: float # might be useless check where it's used
    audio_path: Optional[str]
    languages: List[str]
    tolerance: float

class TranscriptionReviewRequest(BaseModel):
    run_id: str
    transcription: ASRResponse

class AlignmentReviewRequest(BaseModel):
    run_id: str
    alignment: ASRResponse

class TTSReviewSegmentUpdate(BaseModel):
    segment_id: str
    text: str
    lang: Optional[str] = None

class TTSReviewRequest(BaseModel):
    run_id: str
    language: Optional[str] = None
    segments: List[TTSReviewSegmentUpdate] = Field(default_factory=list)

class TTSRegenerateRequest(BaseModel):
    run_id: str
    language: Optional[str] = None
    segment_id: str
    text: str
    lang: Optional[str] = None
    audio_prompt_url: Optional[str] = None
