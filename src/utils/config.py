# simple_voice_chat/config.py
from pathlib import Path

from openai import OpenAI  # For client type hints
from pydantic import BaseModel, Field

# --- Application Constants ---
APP_VERSION = "4.2.0"
OPENAI_TTS_PRICING = {
    # price per 1M token
    "tts-1": 15.00,
    "tts-1-hd": 30.00,
}
OPENAI_TTS_VOICES = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash",
]
# soruce: https://platform.openai.com/docs/pricing
OPENAI_REALTIME_VOICES = OPENAI_TTS_VOICES
OPENAI_REALTIME_PRICING = {
    # Prices per 1M token
    "gpt-4o-mini-realtime-preview-2024-12-17": {
        "input": 10.0,
        "cached_input": 0.30,
        "output": 20.0,
    },
    "gpt-4o-mini-realtime-preview": {
        "input": 10.0,
        "cached_input": 0.30,
        "output": 20.0,
    },
    "gpt-4o-realtime-preview": {
        "input": 40.0,
        "cached_input": 2.50,
        "output": 80.0,
    },
    "gpt-4o-realtime-preview-2024-12-17": {
        "input": 40.0,
        "cached_input": 2.50,
        "output": 80.0,
    },
}
OPENAI_REALTIME_MODELS = list(OPENAI_REALTIME_PRICING.keys())

# gemini backend is not yet implemented but will soon be
GEMINI_LIVE_PRICING = {
    # price per 1M tokens
    # Audio is tokenized at 32 tokens per second.
    # Pricing based on Gemini 1.5 Flash input/output token costs as a proxy, as per https://ai.google.dev/gemini-api/docs/pricing
    "input_text_tokens": 0.35,
    "output_text_tokens": 1.50,
    "cached_text": 0.025,
    "cached_audio": 0.175,
    "input_audio_tokens": 2.10,  # For STT (input audio tokens)
    "output_audio_tokens": 8.5,  # For TTS (output audio tokens)
}
# only one supported model so far
# GEMINI_LIVE_MODELS = ["gemini-2.0-flash-exp"] # TODO: Potentially update if more models supporting LiveConnect become available
GEMINI_LIVE_MODELS = ["gemini-2.0-flash-live-001"]  # TODO: Potentially update if more models supporting LiveConnect become available
GEMINI_LIVE_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]  # From Gemini Live API docs May 2025

# --- End Application Constants ---

# Import env var for Pydantic default
from .env import (
    GEMINI_MODEL_ENV,
    GEMINI_VOICE_ENV,  # Add disable heartbeat env var
    OPENAI_REALTIME_MODEL_ENV,
)


class AppSettings(BaseModel):
    """Centralized application settings.
    This object will be populated at startup and used throughout the application.
    """
    # --- Paths ---
    # These are set during startup in main()
    app_log_dir: Path | None = None
    chat_log_dir: Path | None = None
    tts_base_dir: Path | None = None  # Base dir for all TTS runs
    tts_audio_dir: Path | None = None  # Specific dir for current run's TTS audio

    # --- General App Config ---
    # Populated from args/env in main()
    host: str = "127.0.0.1"
    port: int = 7860  # This will be the actual port used, after checking availability
    preferred_port: int = 7860  # Port from args/env
    verbose: bool = False
    browser: bool = False
    system_message: str = ""
    startup_timestamp_str: str | None = None  # For log filenames etc.
    backend: str = "classic"
    disable_heartbeat: bool = False  # Populated from CLI/env
    openai_realtime_model_arg: str = OPENAI_REALTIME_MODEL_ENV
    # Add after openai_realtime_model_arg

    # --- Gemini Backend Config ---
    gemini_api_key: str | None = None  # Populated from args/env
    gemini_model_arg: str = GEMINI_MODEL_ENV  # Initial preference from args/env
    gemini_voice_arg: str | None = GEMINI_VOICE_ENV  # Initial preference for Gemini voice
    current_gemini_voice: str | None = None  # Actual current voice for Gemini backend
    gemini_context_window_compression_threshold: int = 16000  # Populated from CLI/env, actual value used

    # --- LLM Config (Classic Backend) ---
    # Populated from args/env and derived in main()
    llm_host_arg: str | None = None
    llm_port_arg: str | None = None
    llm_model_arg: str | None = None  # Initial preference for classic backend
    llm_api_key: str | None = None  # For classic backend LLM

    llm_api_base: str | None = None
    use_llm_proxy: bool = False
    current_llm_model: str | None = None  # Actual current model (classic or OpenAI backend model name)

    available_models: list[str] = Field(default_factory=list)  # For classic backend model dropdown
    model_cost_data: dict[str, dict[str, float]] = Field(default_factory=dict)  # For classic backend

    # --- STT Config ---
    # STT parameters are primarily for the 'classic' backend.
    # 'current_stt_language' is also used by the 'openai' backend.
    stt_host_arg: str = "api.openai.com"
    stt_port_arg: str = "443"
    stt_model_arg: str = "whisper-1"
    stt_language_arg: str | None = None  # Initial preference
    stt_api_key: str | None = None  # For classic backend STT

    stt_api_base: str | None = None  # Derived for classic backend
    is_openai_stt: bool = False  # Derived for classic backend
    current_stt_language: str | None = None  # Actual current STT language (both backends)

    # STT Confidence (Classic Backend)
    stt_no_speech_prob_threshold: float = 0.6
    stt_avg_logprob_threshold: float = -0.7
    stt_min_words_threshold: int = 5

    # --- TTS Config (Classic Backend) ---
    # TTS parameters are for the 'classic' backend.
    tts_host_arg: str = "api.openai.com"
    tts_port_arg: str = "443"
    tts_model_arg: str = "tts-1"
    tts_voice_arg: str | None = None  # Initial preference for classic backend TTS
    tts_api_key: str | None = None  # For classic backend TTS
    tts_speed_arg: float = 1.0  # Initial preference
    tts_acronym_preserve_list_arg: str = ""

    tts_base_url: str | None = None  # Derived for classic backend
    is_openai_tts: bool = False  # Derived for classic backend
    tts_acronym_preserve_set: set[str] = Field(default_factory=set)  # Derived for classic backend
    current_tts_voice: str | None = None  # Actual current TTS voice (classic backend)
    current_tts_speed: float = 1.0  # Actual current TTS speed (classic backend)

    available_voices_tts: list[str] = Field(default_factory=list)  # For classic backend voice dropdown OR OpenAI backend

    # --- OpenAI Backend Config ---
    openai_api_key: str | None = None  # Dedicated API key for OpenAI backend
    openai_realtime_voice_arg: str | None = None  # Initial preference for OpenAI backend voice
    current_openai_voice: str | None = None  # Actual current voice for OpenAI backend

    # --- Clients (Classic Backend) ---
    # Initialized in main() if backend is 'classic'
    tts_client: OpenAI | None = None
    stt_client: OpenAI | None = None

    class Config:
        arbitrary_types_allowed = True  # To allow OpenAI client type


# Global instance of settings. This will be populated in main().
# Other modules can import this instance.
settings = AppSettings()

# Export specific constants and the settings instance
__all__ = [
    "APP_VERSION",
    "GEMINI_LIVE_MODELS",  # Add Gemini constant
    "GEMINI_LIVE_PRICING",  # Add Gemini constant
    "GEMINI_LIVE_VOICES",  # Add Gemini constant
    "OPENAI_REALTIME_MODELS",
    "OPENAI_REALTIME_PRICING",
    "OPENAI_REALTIME_VOICES",
    "OPENAI_TTS_PRICING",
    "OPENAI_TTS_VOICES",
    "AppSettings",
    "settings",
]
