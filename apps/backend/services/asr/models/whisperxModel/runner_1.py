import gc
import contextlib
import json
import logging
import os
import ctypes
import importlib
import inspect
import sys
import time
import torch
from pathlib import Path

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


def _patch_hf_hub_download_compat() -> None:
    """
    Compatibility shim for huggingface_hub>=1.0 where hf_hub_download removed
    `use_auth_token` in favor of `token`.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    try:
        sig = inspect.signature(huggingface_hub.hf_hub_download)
    except Exception:
        return

    if "use_auth_token" in sig.parameters:
        return

    original = huggingface_hub.hf_hub_download

    def _compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original(*args, **kwargs)

    huggingface_hub.hf_hub_download = _compat_hf_hub_download  # type: ignore[assignment]
    try:
        import huggingface_hub.file_download as file_download
        file_download.hf_hub_download = _compat_hf_hub_download  # type: ignore[assignment]
    except Exception:
        pass


_patch_hf_hub_download_compat()

import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

# Explicitly fix the whisperx.diarize logger (otherwise it logs to stdout, it is a new issue in latest whisperx)
_diarize_logger = logging.getLogger("whisperx.diarize")
_diarize_logger.handlers = [logging.StreamHandler(sys.stderr)]
_diarize_logger.propagate = False

from common_schemas.models import ASRResponse, Segment, Word
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
        req = ASRResponse(**json.loads(sys.stdin.read()))
        extra = dict(req.extra or {})
        log_level = extra.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level, logging.INFO)

        logger = logging.getLogger("whisperx.runner.align")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False

        with contextlib.redirect_stdout(sys.stderr):
            if not req.audio_url:
                raise ValueError("audio_url is required for alignment.")
            if not req.language:
                raise ValueError("language is required for alignment.")

            req_dict = req.model_dump()
            load_dotenv()
            diarize_enabled = bool(extra.get("enable_diarization", True))
            diarization_model_name = extra.get("diarization_model")
            min_speakers = extra.get("min_speakers")
            max_speakers = extra.get("max_speakers")

            device = _select_device(logger)
            if device == "cpu":
                logger.info("Running alignment on CPU.")

            logger.info(
                "Starting alignment for audio=%s language=%s diarize=%s min_speakers=%s max_speakers=%s",
                req.audio_url,
                req.language,
                diarize_enabled,
                min_speakers,
                max_speakers,
            )

            audio = whisperx.load_audio(req.audio_url)

            align_start = time.perf_counter()
            model_a, metadata = whisperx.load_align_model(
                language_code=req.language,
                device=device,
            )
            logger.info("Loaded alignment model in %.2fs.", time.perf_counter() - align_start)

            align_compute_start = time.perf_counter()
            result = whisperx.align(
                req_dict["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            logger.info("Alignment completed in %.2fs.", time.perf_counter() - align_compute_start)

            diarize_segments = None
            if diarize_enabled and diarization_model_name:
                hf_token = os.getenv("HF_TOKEN")
                diarize_start = time.perf_counter()
                diarize_model = DiarizationPipeline(
                    model_name=diarization_model_name,
                    use_auth_token=hf_token,
                    device=device,
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                logger.info(
                    "Diarization produced %d segments in %.2fs.",
                    len(diarize_segments),
                    time.perf_counter() - diarize_start,
                )
                del diarize_model
                _clear_cuda_cache()
            elif diarize_enabled:
                logger.info("Diarization requested but no model configured; skipping speaker attribution.")
            else:
                logger.info("Diarization disabled for this run; skipping speaker attribution.")

            if diarize_segments is not None:
                result = whisperx.assign_word_speakers(diarize_segments, result)

            segments_out: list[Segment] = convert_whisperx_result_to_Segment(result)
            word_segments_out: list[Word] = create_word_segments(result, segments_out)

            out = ASRResponse(
                segments=segments_out,
                WordSegments=word_segments_out or None,
                language=req.language,
                audio_url=req.audio_url,
                extra=extra,
            )

            del audio
            del model_a
            gc.collect()
            _clear_cuda_cache()

        sys.stdout.write(json.dumps(out.model_dump(), indent=2) + "\n")
        sys.stdout.flush()

    except Exception as exc:
        error_data = {"error": str(exc), "type": type(exc).__name__}
        sys.stderr.write(f"❌ ASR Runner Error: {json.dumps(error_data, indent=2)}\n")
        sys.exit(1)
