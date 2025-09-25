# ComfyUI-VoxCPMTTS
# A clean, efficient ComfyUI custom node for VoxCPM TTS (Text-to-Speech) functionality. This implementation provides high-quality speech generation and voice cloning capabilities using the VoxCPM model.
#
# Models License Notice:
# - VoxCPM-0.5B: Apache-2.0 License (https://huggingface.co/openbmb/VoxCPM-0.5B)
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
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM

_voxcpm_model = None
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

def load_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        models_dir = folder_paths.models_dir
        tts_dir = os.path.join(models_dir, "TTS")
        voxcpm_dir = os.path.join(tts_dir, "VoxCPM-0.5B")
        
        if not os.path.exists(voxcpm_dir) or not os.path.exists(os.path.join(voxcpm_dir, "pytorch_model.bin")):
            try:
                os.makedirs(tts_dir, exist_ok=True)
                os.makedirs(voxcpm_dir, exist_ok=True)
                
                snapshot_download(
                    repo_id="openbmb/VoxCPM-0.5B",
                    local_dir=voxcpm_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"[ERROR] Failed to download VoxCPM model: {str(e)}")
                raise e
        
        try:
            _voxcpm_model = VoxCPM(voxcpm_model_path=voxcpm_dir)
        except Exception as e:
            print(f"[ERROR] VoxCPM initialization failed: {str(e)}")
            raise e
    return _voxcpm_model

class AILab_VoxCPMTTS:
    @classmethod
    def INPUT_TYPES(s):
        available_devices = get_available_devices()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is VoxCPM TTS.", "tooltip": "Text to synthesize into speech"}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Guidance scale: higher = more adherence to prompt, lower = more natural"}),
                "inference_steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of diffusion steps: higher = better quality, lower = faster"}),
                "max_length": ("INT", {"default": 4096, "min": 256, "max": 8192, "step": 256, "tooltip": "Maximum token length during generation"}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Enable text normalization for numbers, punctuation, etc."}),
                "device": (available_devices, {"default": available_devices[0], "tooltip": "Device to run the model on"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Random seed (-1 for random)"}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio for voice cloning"}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text transcript of reference audio (leave empty for auto-transcription)"}),
                "fade_in_ms": ("INT", {"default": 20, "min": 0, "max": 1000, "step": 5, "tooltip": "Fade-in duration to reduce initial artifacts (0-1000ms)"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("REFERENCE_TEXT", "AUDIO")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🔊TTS/VoxCPMTTS"
    
    def generate(self, text, cfg_value, inference_steps, max_length, normalize, device, seed, reference_audio=None, reference_text="", fade_in_ms=20):
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        model_management.soft_empty_cache()
        model = load_model()
        
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

            if isinstance(reference_text, str) and len(reference_text.strip()) >= 3:
                prompt_text = reference_text.strip()
            else:
                tmp_wav = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        torchaudio.save(tmp_file.name, prompt_waveform.squeeze(0), prompt_sample_rate)
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

        # silent
            
        wav = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_waveform=prompt_waveform,
            prompt_sample_rate=prompt_sample_rate,
            prompt_text=prompt_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_steps),
            max_length=max_length,
            normalize=normalize,
            retry_badcase=True,
            retry_badcase_max_times=2,
            retry_badcase_ratio_threshold=8.0,
        )
        # no temp files used
    
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
            sr = 16000
            n = int(max(0, fade_in_ms) * sr / 1000)
            if n > 0 and audio_tensor.shape[-1] > n:
                ramp = torch.linspace(0.0, 1.0, n, dtype=audio_tensor.dtype, device=audio_tensor.device)
                audio_tensor[..., :n] *= ramp
        except Exception:
            pass
        
        used_reference_text = prompt_text if isinstance(prompt_text, str) else ""
        return (used_reference_text, {"waveform": audio_tensor, "sample_rate": 16000})

NODE_CLASS_MAPPINGS = {"AILab_VoxCPMTTS": AILab_VoxCPMTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"AILab_VoxCPMTTS": "VoxCPM TTS"}