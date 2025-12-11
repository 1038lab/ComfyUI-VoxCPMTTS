# ComfyUI-VoxCPMTTS
# A clean, efficient ComfyUI custom node for VoxCPM TTS (Text-to-Speech) functionality. This implementation provides high-quality speech generation and voice cloning capabilities using the VoxCPM model.
#
# Models License Notice:
# - VoxCPM1.5: Apache-2.0 License (https://huggingface.co/openbmb/VoxCPM1.5)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-VoxCPMTTS

import torch
import numpy as np
import os
import folder_paths
import comfy.model_management as model_management
import soundfile as sf
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM

DEFAULT_MODEL_ID = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM1.5")
DEFAULT_MODEL_DIRNAME = os.getenv("VOXCPM_MODEL_DIR", DEFAULT_MODEL_ID.split("/")[-1])
MODEL_CHOICES = ["VoxCPM 1.5", "VoxCPM 0.5B"]
MODEL_REPO_MAP = {
    "VoxCPM 1.5": "openbmb/VoxCPM1.5",
    "VoxCPM 0.5B": "openbmb/VoxCPM-0.5B",
}
DEFAULT_MODEL_CHOICE = next((name for name, repo in MODEL_REPO_MAP.items() if repo == DEFAULT_MODEL_ID and name in MODEL_CHOICES), MODEL_CHOICES[0])

_voxcpm_models = {}
_asr_model = None
_asr_backend = None

def get_available_devices():
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices

def get_best_device():
    if torch.cuda.is_available():
        return model_management.get_torch_device()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def unload_model_instance(repo_id: str):
    global _voxcpm_models
    to_delete = []
    for (rid, _), mdl in _voxcpm_models.items():
        if rid == repo_id:
            try:
                if hasattr(mdl, "tts_model"):
                    mdl.tts_model.to(torch.device("cpu"))
            except Exception:
                pass
            to_delete.append((rid, _))
    for key in to_delete:
        _voxcpm_models.pop(key, None)
    model_management.soft_empty_cache()

def load_model(model_repo: str = DEFAULT_MODEL_ID, model_dirname: str = None):
    global _voxcpm_models
    repo_id = model_repo or DEFAULT_MODEL_ID
    dirname = model_dirname or os.getenv("VOXCPM_MODEL_DIR") or repo_id.split("/")[-1]
    models_dir = folder_paths.models_dir
    tts_dir = os.path.join(models_dir, "TTS")
    target_dir = os.path.join(tts_dir, dirname)
    cache_key = (repo_id, target_dir)

    if cache_key not in _voxcpm_models:
        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            try:
                os.makedirs(tts_dir, exist_ok=True)
                os.makedirs(target_dir, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"[ERROR] Failed to download VoxCPM model: {str(e)}")
                raise e

        try:
            _voxcpm_models[cache_key] = VoxCPM(
                voxcpm_model_path=target_dir,
                enable_denoiser=False,
                optimize=False,
            )
        except Exception as e:
            print(f"[ERROR] VoxCPM initialization failed: {str(e)}")
            raise e
    return _voxcpm_models[cache_key]

class AILab_VoxCPMTTS:
    @classmethod
    def INPUT_TYPES(s):
        available_devices = get_available_devices()
        return {
            "required": {
                "model": (MODEL_CHOICES, {"default": DEFAULT_MODEL_CHOICE, "tooltip": "Select VoxCPM model version"}),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is VoxCPM TTS.", "tooltip": "Text to synthesize into speech"}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning"}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text transcript of reference audio (leave empty for auto-transcription)"}),
                "auto_transcribe_reference": ("BOOLEAN", {"default": False, "tooltip": "If reference text is empty, auto-run ASR to fill it"}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model after generation to free VRAM"}),
                "device": (available_devices, {"default": available_devices[0], "tooltip": "Device to run the model on"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Random seed (-1 for random)"}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("REFERENCE_TEXT", "AUDIO")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🔊TTS/VoxCPMTTS"
    
    def generate(self, model, text, device, seed, reference_audio=None, reference_text="", auto_transcribe_reference=False, unload_model=False):
        # Hidden presets for simplified node
        cfg_value = 2.0
        inference_steps = 10
        max_length = 4096
        fade_in_ms = 20
        retry_attempts = 2
        retry_threshold = 8.0
        normalize = True
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        model_management.soft_empty_cache()
        repo_id = MODEL_REPO_MAP.get(model, MODEL_REPO_MAP.get(model.strip(), model))
        model = load_model(model_repo=repo_id)
        target_sr = 44100 if "1.5" in repo_id else 16000
        
        if device == "auto":
            target_device = get_best_device()
        elif device == "cuda" and torch.cuda.is_available():
            target_device = model_management.get_torch_device()
        else:
            target_device = torch.device("cpu")
        
        try:
            if hasattr(model, 'tts_model'):
                model.tts_model.to(target_device)
                if hasattr(model.tts_model, 'audio_vae'):
                    model.tts_model.audio_vae.to(target_device)
        except Exception as e:
            pass
        
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = None
        prompt_waveform = None
        prompt_sample_rate = None
        prompt_text = None

        if reference_audio:
            import torchaudio, tempfile
            waveform = reference_audio["waveform"]
            sample_rate = reference_audio["sample_rate"]
            # Normalize to shape [1,1,T]
            while waveform.dim() > 3:
                waveform = waveform.squeeze(0)
            if waveform.dim() == 3:
                if waveform.shape[0] > 1:
                    waveform = waveform[0:1, ...]
            elif waveform.dim() == 2:
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            else:
                waveform = waveform.reshape(1, 1, -1)
            prompt_waveform = waveform.contiguous().float().cpu()
            prompt_sample_rate = int(sample_rate)

            if isinstance(reference_text, str) and len(reference_text.strip()) >= 1:
                prompt_text = reference_text.strip()
            elif auto_transcribe_reference:
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        wav_np = prompt_waveform.squeeze().cpu().numpy()
                        if wav_np.ndim > 1:
                            wav_np = wav_np[0]
                        sf.write(tmp_file.name, wav_np, prompt_sample_rate)
                        tmp_wav = tmp_file.name
                    global _asr_model
                    global _asr_backend
                    if _asr_model is None:
                        # Prefer faster-whisper; fallback to openai-whisper
                        try:
                            from faster_whisper import WhisperModel  # type: ignore
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            compute_type = "float16" if torch.cuda.is_available() else "int8"
                            model_name = os.environ.get("VOXCPM_ASR_MODEL", "small")
                            print(f"[ASR] Loading faster-whisper: {model_name} on {device}/{compute_type}")
                            _asr_model = WhisperModel(model_name, device=device, compute_type=compute_type)
                            _asr_backend = "faster_whisper"
                        except Exception:
                            try:
                                import whisper  # type: ignore
                                model_name = os.environ.get("VOXCPM_ASR_MODEL", "small")
                                print(f"[ASR] Loading openai-whisper: {model_name}")
                                _asr_model = whisper.load_model(model_name)
                                _asr_backend = "openai_whisper"
                            except Exception as e_load:
                                print(f"[ASR] Failed to load ASR backends: {e_load}")
                                _asr_backend = None
                    text_out = ""
                    if _asr_backend == "faster_whisper":
                        segments, _ = _asr_model.transcribe(tmp_wav, language=None, beam_size=1)
                        text_out = " ".join([seg.text for seg in segments]).strip()
                    elif _asr_backend == "openai_whisper":
                        res = _asr_model.transcribe(tmp_wav)
                        text_out = (res.get("text", "") or "").strip()
                    prompt_text = text_out if len(text_out) > 0 else None
                    print(f"[ASR] Transcription: {text_out[:120]}" + ("..." if len(text_out) > 120 else ""))
                except Exception as e:
                    prompt_text = None
                    print(f"[ASR] Failed: {e}")
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.unlink(tmp_wav)
                        except Exception:
                            pass
            if prompt_text is None:
                raise ValueError("reference_text is required when providing reference_audio (or enable auto_transcribe_reference and ensure ASR succeeds)")

        tmp_prompt_file = None
        if prompt_wav_path is None and prompt_waveform is not None and prompt_sample_rate is not None:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    wav_np = prompt_waveform.squeeze().cpu().numpy()
                    if wav_np.ndim > 1:
                        wav_np = wav_np[0]
                    sf.write(tmp_file.name, wav_np, prompt_sample_rate)
                    tmp_prompt_file = tmp_file.name
                    prompt_wav_path = tmp_prompt_file
            except Exception as e:
                print(f"[WARN] Failed to write prompt waveform: {e}")
                prompt_wav_path = None

        try:
            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_steps),
                max_len=max_length,
                normalize=normalize,
                retry_badcase=retry_attempts > 0,
                retry_badcase_max_times=retry_attempts,
                retry_badcase_ratio_threshold=retry_threshold,
            )
        finally:
            if tmp_prompt_file and os.path.exists(tmp_prompt_file):
                try:
                    os.unlink(tmp_prompt_file)
                except Exception:
                    pass
    
        if isinstance(wav, np.ndarray):
            audio_tensor = torch.from_numpy(wav).float()
        else:
            audio_tensor = wav.float()
        
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        try:
            n = int(max(0, fade_in_ms) * target_sr / 1000)
            if n > 0 and audio_tensor.shape[-1] > n:
                ramp = torch.linspace(0.0, 1.0, n, dtype=audio_tensor.dtype, device=audio_tensor.device)
                audio_tensor[..., :n] *= ramp
        except Exception:
            pass
        
        used_reference_text = prompt_text if isinstance(prompt_text, str) else ""
        result = (used_reference_text, {"waveform": audio_tensor, "sample_rate": target_sr})

        if unload_model:
            unload_model_instance(repo_id)

        return result

class AILab_VoxCPMTTS_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        available_devices = get_available_devices()
        return {
            "required": {
                "model": (MODEL_CHOICES, {"default": DEFAULT_MODEL_CHOICE, "tooltip": "Select VoxCPM model version"}),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is VoxCPM TTS.", "tooltip": "Text to synthesize into speech"}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Guidance scale: higher = more adherence to prompt, lower = more natural"}),
                "inference_steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of diffusion steps: higher = better quality, lower = faster"}),
                "max_length": ("INT", {"default": 4096, "min": 256, "max": 8192, "step": 256, "tooltip": "Maximum token length during generation"}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Enable text normalization for numbers, punctuation, etc."}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning"}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text transcript of reference audio (leave empty for auto-transcription)"}),
                "fade_in_ms": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 5, "tooltip": "Fade-in duration to reduce initial artifacts (0-1000ms)"}),
                "auto_transcribe_reference": ("BOOLEAN", {"default": False, "tooltip": "Automatically transcribe reference audio when no text is provided"}),
                "show_transcription_log": ("BOOLEAN", {"default": True, "tooltip": "Show ASR transcription logs in console"}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model after generation to free VRAM"}),
                "retry_attempts": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1, "tooltip": "Number of retry attempts for bad cases"}),
                "retry_threshold": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.1, "tooltip": "Audio-to-text ratio threshold for retry detection"}),
                "device": (available_devices, {"default": available_devices[0], "tooltip": "Device to run the model on"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Random seed (-1 for random)"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🔊TTS/VoxCPMTTS"
    
    def generate(
        self,
        model,
        text,
        cfg_value,
        inference_steps,
        max_length,
        normalize,
        seed,
        device,
        retry_attempts,
        retry_threshold,
        reference_audio=None,
        reference_text="",
        fade_in_ms=20,
        auto_transcribe_reference=False,
        show_transcription_log=True,
        unload_model=False,
    ):
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        model_management.soft_empty_cache()
        repo_id = MODEL_REPO_MAP.get(model, MODEL_REPO_MAP.get(model.strip(), model))
        model = load_model(model_repo=repo_id)
        target_sr = 44100 if "1.5" in repo_id else 16000
        
        if device == "auto":
            target_device = get_best_device()
        elif device == "cuda" and torch.cuda.is_available():
            target_device = model_management.get_torch_device()
        else:
            target_device = torch.device("cpu")
        
        try:
            if hasattr(model, 'tts_model'):
                model.tts_model.to(target_device)
                if hasattr(model.tts_model, 'audio_vae'):
                    model.tts_model.audio_vae.to(target_device)
        except Exception:
            pass
        
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = None
        prompt_waveform = None
        prompt_sample_rate = None
        prompt_text = None
        
        if reference_audio:
            import torchaudio, tempfile
            waveform = reference_audio["waveform"]
            sample_rate = reference_audio["sample_rate"]
            while waveform.dim() > 3:
                waveform = waveform.squeeze(0)
            if waveform.dim() == 3:
                if waveform.shape[0] > 1:
                    waveform = waveform[0:1, ...]
            elif waveform.dim() == 2:
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            else:
                waveform = waveform.reshape(1, 1, -1)
            prompt_waveform = waveform.contiguous().float().cpu()
            prompt_sample_rate = int(sample_rate)
            if isinstance(reference_text, str) and len(reference_text.strip()) >= 1:
                prompt_text = reference_text.strip()
            elif auto_transcribe_reference:
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        wav_np = prompt_waveform.squeeze().cpu().numpy()
                        if wav_np.ndim > 1:
                            wav_np = wav_np[0]
                        sf.write(tmp_file.name, wav_np, prompt_sample_rate)
                        tmp_wav = tmp_file.name
                    global _asr_model, _asr_backend
                    if _asr_model is None:
                        try:
                            from faster_whisper import WhisperModel  # type: ignore
                            asr_device = "cuda" if torch.cuda.is_available() else "cpu"
                            compute_type = "float16" if torch.cuda.is_available() else "int8"
                            model_name = os.environ.get("VOXCPM_ASR_MODEL", "small")
                            if show_transcription_log:
                                print(f"[ASR] Loading faster-whisper: {model_name} on {asr_device}/{compute_type}")
                            _asr_model = WhisperModel(model_name, device=asr_device, compute_type=compute_type)
                            _asr_backend = "faster_whisper"
                        except Exception:
                            try:
                                import whisper  # type: ignore
                                model_name = os.environ.get("VOXCPM_ASR_MODEL", "small")
                                if show_transcription_log:
                                    print(f"[ASR] Loading openai-whisper: {model_name}")
                                _asr_model = whisper.load_model(model_name)
                                _asr_backend = "openai_whisper"
                            except Exception as e_load:
                                if show_transcription_log:
                                    print(f"[ASR] Failed to load ASR backends: {e_load}")
                                _asr_backend = None
                    text_out = ""
                    if _asr_backend == "faster_whisper":
                        segments, _ = _asr_model.transcribe(tmp_wav, language=None, beam_size=1)
                        text_out = " ".join([seg.text for seg in segments]).strip()
                    elif _asr_backend == "openai_whisper":
                        res = _asr_model.transcribe(tmp_wav)
                        text_out = (res.get("text", "") or "").strip()
                    prompt_text = text_out if len(text_out) > 0 else None
                    if show_transcription_log:
                        print(f"[ASR] Transcription: {text_out[:120]}" + ("..." if len(text_out) > 120 else ""))
                except Exception as e:
                    prompt_text = None
                    if show_transcription_log:
                        print(f"[ASR] Failed: {e}")
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.unlink(tmp_wav)
                        except Exception:
                            pass
            if prompt_text is None:
                if show_transcription_log:
                    print("[ERROR] reference_text is required when providing reference_audio (or enable auto_transcribe_reference and ensure ASR succeeds)")
                raise ValueError("reference_text is required when providing reference_audio (or enable auto_transcribe_reference and ensure ASR succeeds)")
            
        tmp_prompt_file = None
        if prompt_wav_path is None and prompt_waveform is not None and prompt_sample_rate is not None:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    wav_np = prompt_waveform.squeeze().cpu().numpy()
                    if wav_np.ndim > 1:
                        wav_np = wav_np[0]
                    sf.write(tmp_file.name, wav_np, prompt_sample_rate)
                    tmp_prompt_file = tmp_file.name
                    prompt_wav_path = tmp_prompt_file
            except Exception as e:
                if show_transcription_log:
                    print(f"[WARN] Failed to write prompt waveform: {e}")
                prompt_wav_path = None

        try:
            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_steps),
                max_len=max_length,
                normalize=normalize,
                retry_badcase=retry_attempts > 0,
                retry_badcase_max_times=retry_attempts,
                retry_badcase_ratio_threshold=retry_threshold,
            )
        finally:
            if tmp_prompt_file and os.path.exists(tmp_prompt_file):
                try:
                    os.unlink(tmp_prompt_file)
                except Exception:
                    pass
    
        if isinstance(wav, np.ndarray):
            audio_tensor = torch.from_numpy(wav).float()
        else:
            audio_tensor = wav.float()
        
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        try:
            n = int(max(0, fade_in_ms) * target_sr / 1000)
            if n > 0 and audio_tensor.shape[-1] > n:
                ramp = torch.linspace(0.0, 1.0, n, dtype=audio_tensor.dtype, device=audio_tensor.device)
                audio_tensor[..., :n] *= ramp
        except Exception:
            pass
        
        used_reference_text = prompt_text if isinstance(prompt_text, str) else ""
        result = ({"waveform": audio_tensor, "sample_rate": target_sr}, used_reference_text)

        if unload_model:
            unload_model_instance(repo_id)

        return result

NODE_CLASS_MAPPINGS = {
    "AILab_VoxCPMTTS": AILab_VoxCPMTTS,
    "AILab_VoxCPMTTS_Advanced": AILab_VoxCPMTTS_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_VoxCPMTTS": "VoxCPM TTS",
    "AILab_VoxCPMTTS_Advanced": "VoxCPM TTS (Advanced)",
}
