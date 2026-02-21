import contextlib
import gc
import json
import logging
import os
import ctypes
import importlib
import sys
import time
from pathlib import Path

import torch

# PyTorch 2.6 changed weights_only default to True; pyannote checkpoints use
# omegaconf types that must be explicitly allowlisted before any model is loaded.
# We trust these model artifacts (HuggingFace / pyannote), so we also opt out of
# weights-only mode for this worker to preserve compatibility.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata

    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
except Exception:
    pass

import whisperx
from dotenv import load_dotenv

from common_schemas.models import ASRRequest, ASRResponse, Segment, Word
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments

_CUDNN_CANDIDATES = (
    # cuDNN 9.x
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    # cuDNN 8.x
    "libcudnn_ops_infer.so.8",
    "libcudnn_cnn_infer.so.8",
    "libcudnn.so.8",
)
_NVIDIA_LIB_MODULES = (
    "nvidia.cudnn.lib",
    "nvidia.cublas.lib",
    "nvidia.cuda_runtime.lib",
    "nvidia.cufft.lib",
    "nvidia.curand.lib",
    "nvidia.cusolver.lib",
    "nvidia.cusparse.lib",
    "nvidia.nvjitlink.lib",
)


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    for module_name in _NVIDIA_LIB_MODULES:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        mod_file = getattr(mod, "__file__", "")
        if not mod_file:
            continue
        lib_dir = Path(mod_file).resolve().parent
        if lib_dir.exists() and lib_dir not in dirs:
            dirs.append(lib_dir)
    return dirs


def _prepend_ld_library_path(lib_dirs: list[Path]) -> None:
    if not lib_dirs:
        return
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    prefixes = [str(p) for p in lib_dirs if str(p) not in existing]
    if prefixes:
        os.environ["LD_LIBRARY_PATH"] = ":".join(prefixes + existing)


def _load_cudnn_runtime(logger: logging.Logger) -> str | None:
    errors: list[str] = []

    for lib_name in _CUDNN_CANDIDATES:
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            return lib_name
        except OSError as exc:
            errors.append(f"{lib_name}: {exc}")

    lib_dirs = _nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)
    for lib_dir in lib_dirs:
        for lib_name in _CUDNN_CANDIDATES:
            lib_path = lib_dir / lib_name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                return lib_name
            except OSError as exc:
                errors.append(f"{lib_path}: {exc}")

    logger.warning(
        "CUDA is available but no compatible cuDNN runtime was found (%s). Falling back to CPU.",
        "; ".join(errors[-3:]) if errors else "no candidate libraries found",
    )
    return None


def _select_device(logger: logging.Logger) -> str:
    """Prefer CUDA only when required cuDNN runtime libs are available."""
    if not torch.cuda.is_available():
        return "cpu"
    loaded = _load_cudnn_runtime(logger)
    if loaded:
        logger.info("Using CUDA with %s", loaded)
        return "cuda"
    return "cpu"


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
            
            whisper_model = extra.get("model_name", "large")
            batch_size = extra.get("batch_size", 16)  # reduce if low on GPU mem
            compute_type = extra.get("compute_type", "float16")  # change to "int8" if low on GPU mem (may reduce accuracy)

            load_dotenv()

            device = _select_device(logger)
            if device == "cpu" and compute_type not in {"int8", "float32"}:
                logger.info("Switching compute_type from %s to float32 for CPU inference.", compute_type)
                compute_type = "float32"
            if device == "cpu":
                batch_size = min(batch_size, 4)

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
            try:
                if req.language_hint:
                    logger.info("Transcribing with language hint=%s.", req.language_hint)
                    result_0 = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
                else:
                    logger.info("Transcribing with automatic language detection.")
                    result_0 = model.transcribe(audio, batch_size=batch_size)
            except IndexError as exc:
                # WhisperX/pyannote can raise IndexError when VAD finds no active
                # speech in the input. Treat this as a valid empty transcription.
                logger.warning(
                    "ASR produced no active speech (%s). Returning empty segments.",
                    exc,
                )
                result_0 = {
                    "segments": [],
                    "word_segments": [],
                    "language": req.language_hint,
                }
            except Exception as exc:
                # Defensive fallback in case CUDA passes preflight but runtime still
                # fails due missing/invalid cuDNN libs.
                if device == "cuda" and "cudnn" in str(exc).lower():
                    logger.warning(
                        "CUDA transcription failed due cuDNN runtime issue (%s). Retrying on CPU.",
                        exc,
                    )
                    del model
                    gc.collect()
                    _clear_cuda_cache()
                    device = "cpu"
                    compute_type = "float32"
                    batch_size = min(batch_size, 4)
                    model = whisperx.load_model(
                        whisper_model,
                        device,
                        compute_type=compute_type,
                        download_root=str(model_dir),
                    )
                    if req.language_hint:
                        result_0 = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
                    else:
                        result_0 = model.transcribe(audio, batch_size=batch_size)
                else:
                    raise
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
        error_data = {"error": str(e), "type": type(e).__name__}
        sys.stderr.write(f"❌ ASR Runner Error: {json.dumps(error_data, indent=2)}\n")
        sys.exit(1)
