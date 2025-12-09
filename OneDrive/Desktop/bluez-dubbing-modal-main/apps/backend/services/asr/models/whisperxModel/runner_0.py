import os
import contextlib
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import whisperx
from dotenv import load_dotenv

from common_schemas.models import ASRRequest, ASRResponse, Segment, Word
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


import traceback

if __name__ == "__main__":
    try:
        BASE = Path(__file__).resolve().parents[4]
        req = ASRRequest(**json.loads(sys.stdin.read()))
        extra = dict(req.extra or {})
        log_level = extra.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level, logging.INFO)

        logger = logging.getLogger("whisperx.runner")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False

        run_start = time.perf_counter()

        with contextlib.redirect_stdout(sys.stderr):
            
            whisper_model = extra.get("model_name") or os.getenv("WHISPERX_MODEL_NAME", "small")
            batch_size    = int(os.getenv("WHISPERX_BATCH_SIZE", extra.get("batch_size", 1)))
            compute_type  = extra.get("compute_type") or os.getenv("WHISPERX_COMPUTE_TYPE", "float32")

            load_dotenv()

            device = extra.get("device") or os.getenv("WHISPERX_DEVICE") \
                     or ("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(
                "Starting transcription run for audio=%s model=%s device=%s batch_size=%s compute_type=%s",
                req.audio_url,
                whisper_model,
                device,
                batch_size,
                compute_type,
            )

            model_dir = BASE / "models_cache" / "asr"

            load_start = time.perf_counter()
            model = whisperx.load_model(whisper_model, device, compute_type=compute_type, download_root=str(model_dir))
            logger.info("Loaded WhisperX model in %.2fs.", time.perf_counter() - load_start)

            audio_load_start = time.perf_counter()
            audio = whisperx.load_audio(req.audio_url)
            logger.info("Loaded audio in %.2fs.", time.perf_counter() - audio_load_start)

            transcribe_start = time.perf_counter()
            if req.language_hint:
                logger.info("Transcribing with language hint=%s.", req.language_hint)
                result_0 = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
            else:
                logger.info("Transcribing with automatic language detection.")
                result_0 = model.transcribe(audio, batch_size=batch_size)
            logger.info(
                "Transcription finished in %.2fs (segments=%d).",
                time.perf_counter() - transcribe_start,
                len(result_0.get("segments", [])),
            )

            language = result_0.get("language")
            logger.info("Detected language=%s.", language)

            # release model and audio to keep VRAM usage low
            del audio
            gc.collect()
            _clear_cuda_cache()
            del model

            raw_segments_out: list[Segment] = convert_whisperx_result_to_Segment(result_0)
            raw_word_segments_out: list[Word] = create_word_segments(result_0, raw_segments_out)

            raw_output = ASRResponse(
                segments=raw_segments_out,
                WordSegments=raw_word_segments_out or None,
                language=language,
                audio_url=req.audio_url,
                extra={
                    **extra,
                    "min_speakers": req.min_speakers,
                    "max_speakers": req.max_speakers,
                },
            )

            logger.info(
                "Completed transcription. segments=%d language=%s runtime=%.2fs",
                len(raw_segments_out),
                language,
                time.perf_counter() - run_start,
            )

        sys.stdout.write(json.dumps(raw_output.model_dump(), indent=2) + "\n")
        sys.stdout.flush()

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        error_data = {"error": str(e), "type": type(e).__name__}
        sys.stderr.write(f"❌ ASR Runner Error: {json.dumps(error_data, indent=2)}\n")
        sys.exit(1)
