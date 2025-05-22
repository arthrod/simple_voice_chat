import json
import os # Ensure os is imported early
import random

# Set LITELLM_MODE to PRODUCTION before litellm is imported
os.environ["LITELLM_MODE"] = "PRODUCTION"
import asyncio
import base64 # Ensure it's used if needed by OpenAIRealtimeHandler
import datetime
import re
import threading
import time
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import click # Added click
import litellm
import numpy as np
import openai # Ensure openai is imported for AsyncOpenAI
import platformdirs
import uvicorn
import webview
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    AsyncStreamHandler, # Added AsyncStreamHandler
    ReplyOnPause,
    SileroVadOptions,
    Stream,
    get_twilio_turn_credentials,
    wait_for_item, # Added wait_for_item
)
from google import genai
from google.genai.types import (
    ContextWindowCompressionConfig,
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SlidingWindow,
    Content, 
    Part,    
    Blob,    
    FunctionResponse, 
    ToolCall, # Explicitly import ToolCall if it's a distinct type used
    # TODO: If specific RecognitionConfig for STT language is found for google-generativeai, import it.
    # from google.ai import generativelanguage as glm -> this requires google-cloud-aiplatform
)
from google.genai.types import (
    SpeechConfig as GenaiSpeechConfig,
)
from google.genai.types import (
    VoiceConfig as GenaiVoiceConfig,
)
from gradio.utils import get_space
from loguru import logger
from openai import AuthenticationError, OpenAI 
from pydantic import BaseModel
from pydub import AudioSegment

# --- Import Configuration ---
from src.utils.config import (
    APP_VERSION,
    GEMINI_LIVE_MODELS, 
    GEMINI_LIVE_PRICING, 
    GEMINI_LIVE_VOICES, 
    OPENAI_REALTIME_MODELS, 
    OPENAI_REALTIME_PRICING, 
    OPENAI_REALTIME_VOICES, 
    OPENAI_TTS_PRICING,
    OPENAI_TTS_VOICES,
    AppSettings, 
    settings,
)

# --- End Import Configuration ---
from src.utils.env import (
    APP_PORT_ENV,
    DEFAULT_LLM_MODEL_ENV,
    DEFAULT_TTS_SPEED_ENV,
    DEFAULT_VOICE_TTS_ENV,

    DISABLE_HEARTBEAT_ENV, 
    GEMINI_API_KEY_ENV, 
    GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD_ENV, 
    GEMINI_MODEL_ENV, 
    GEMINI_VOICE_ENV, 

    DEFAULT_SYSTEM_INSTRUCTION_ENV,
    LLM_API_KEY_ENV,
    LLM_HOST_ENV,
    LLM_PORT_ENV,
    OPENAI_API_KEY_ENV,
    OPENAI_REALTIME_MODEL_ENV,
    OPENAI_REALTIME_VOICE_ENV,
    STT_API_KEY_ENV,
    STT_AVG_LOGPROB_THRESHOLD_ENV,
    STT_HOST_ENV,
    STT_LANGUAGE_ENV,
    STT_MIN_WORDS_THRESHOLD_ENV,
    STT_MODEL_ENV,
    STT_NO_SPEECH_PROB_THRESHOLD_ENV,
    STT_PORT_ENV,
    SYSTEM_MESSAGE_ENV,
    TTS_ACRONYM_PRESERVE_LIST_ENV,
    TTS_API_KEY_ENV,
    TTS_HOST_ENV,
    TTS_MODEL_ENV,
    TTS_PORT_ENV,
)
from src.utils.llms import (
    calculate_llm_cost,
    get_models_and_costs_from_litellm,
    get_models_and_costs_from_proxy,
)
from src.utils.logging_config import setup_logging
from src.utils.misc import is_port_in_use
from src.utils.stt import check_stt_confidence, transcribe_audio
from src.utils.tts import (
    generate_tts_for_sentence,
    get_voices,
    prepare_available_voices_data,
)

# --- Global Variables (Runtime State - Not Configuration) ---
last_heartbeat_time: datetime.datetime | None = None
heartbeat_timeout: int = 15 
shutdown_event = threading.Event() 
pywebview_window = None 
uvicorn_server = None 
form_data = {
    "title": "",
    "description": "",
    "required": [],
    "content": {},
}
# --- End Global Configuration & State ---

# --- Constants ---
OPENAI_REALTIME_SAMPLE_RATE = 24000
GEMINI_REALTIME_INPUT_SAMPLE_RATE = 16000 
GEMINI_REALTIME_OUTPUT_SAMPLE_RATE = 24000 
# --- End Constants ---


def load_form_data() -> None:
    global form_data
    form_path = Path("form.json")
    try:
        if form_path.exists():
            with open(form_path, encoding="utf-8") as f:
                loaded_data = json.load(f)
                global form_data
                form_data = loaded_data
                logger.info(f"Formulário carregado com {len(form_data.get('content', {}))} campos")
        else:
            logger.warning("Arquivo form.json não encontrado, usando formulário vazio")
    except Exception as e:
        logger.error(f"Erro ao carregar dados do formulário: {e}")

def save_chat_history(history: list[dict[str, str]]) -> None:
    if not settings.chat_log_dir or not settings.startup_timestamp_str:
        logger.warning("Chat log directory or startup timestamp not initialized. Cannot save history.")
        return
    log_file_path = settings.chat_log_dir / f"{settings.startup_timestamp_str}.json"
    logger.debug(f"Saving chat history to: {log_file_path}")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.debug(f"Successfully saved chat history ({len(history)} messages).")
    except OSError as e:
        logger.error(f"Failed to save chat history to {log_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving chat history: {e}")

async def response(audio: tuple[int, np.ndarray], chatbot: list[dict] | None = None) -> AsyncGenerator[Any]:
    # ... (Classic backend response logic - unchanged) ...
    # Access module-level variables set after arg parsing in main()
    # Ensure clients are initialized (should be, but good practice)
    if not settings.stt_client or not settings.tts_client:
        logger.error("STT or TTS client not initialized for classic backend. Cannot process request.")
        # Yield error state?
        return

    # Work with a copy to avoid modifying the input list directly and ensure clean state per call
    current_chatbot = (chatbot or []).copy()
    logger.info(
        f"--- Entering response function with history length: {len(current_chatbot)} ---",
    )
    # Extract only role and content for sending to the LLM API
    # Handle both old dict format and new ChatMessage model format during transition if needed,
    # but current_chatbot should ideally contain ChatMessage objects or dicts matching its structure.
    messages = []
    for item in current_chatbot:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        elif hasattr(item, "role") and hasattr(item, "content"):  # Check if it looks like ChatMessage
             messages.append({"role": item.role, "content": item.content})
        else:
            logger.warning(f"Skipping unexpected item in chatbot history: {item}")

    # Add system message if defined
    if settings.system_message:
        # Prepend system message if not already present (e.g., first turn)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": settings.system_message})
            logger.debug("Prepended system message to LLM input.")
        elif (
            messages[0].get("role") == "system"
            and messages[0].get("content") != settings.system_message
        ):
            # Update system message if it changed (though this shouldn't happen with current setup)
            messages[0]["content"] = settings.system_message
            logger.debug("Updated existing system message in LLM input.")

    # Signal STT processing start
    yield AdditionalOutputs(
        {
            "type": "status_update",
            "status": "stt_processing",
            "message": "Transcribing...",
        },
    )

    # --- Speech-to-Text using imported function ---
    stt_success, prompt, stt_response_obj, stt_error = await transcribe_audio(
        audio,
        settings.stt_client,
        settings.stt_model_arg,  # Use the model name from initial args/env
        settings.current_stt_language,
        settings.stt_api_base,
    )

    if not stt_success:
        logger.error(f"STT failed: {stt_error}")
        stt_error_msg = {
            "role": "assistant",
            "content": f"[STT Error: {stt_error or 'Unknown STT failure'}]",
        }
        yield AdditionalOutputs({"type": "chatbot_update", "message": stt_error_msg})
        # Yield final state and status even on STT error to reset frontend
        logger.warning("Yielding final state after STT error...")
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,
            },  # Yield original state
        )
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (after STT error)",
            },
        )
        logger.info("--- Exiting response function after STT error ---")
        return

    # --- STT Confidence Check using imported function ---
    reject_transcription, rejection_reason = check_stt_confidence(
        stt_response_obj,  # The full response object from STT
        prompt,  # The transcribed text
        settings.stt_no_speech_prob_threshold,
        settings.stt_avg_logprob_threshold,
        settings.stt_min_words_threshold,
    )
    # Store relevant STT details for potential metadata logging
    stt_metadata_details = {}
    if hasattr(stt_response_obj, "no_speech_prob"):
        stt_metadata_details["no_speech_prob"] = stt_response_obj.no_speech_prob
    if hasattr(stt_response_obj, "avg_logprob"):
        stt_metadata_details["avg_logprob"] = stt_response_obj.avg_logprob
    # Add word count if needed: stt_metadata_details['word_count'] = len(prompt.split())

    if reject_transcription:
        logger.warning(f"STT confidence check failed: {rejection_reason}. Details: {stt_metadata_details}")
        # Yield status updates to go back to idle without processing this prompt
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": f"Listening (low confidence: {rejection_reason})",
            },
        )
        # Yield final state (unchanged history)
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,  # Send back original history
            },
        )
        logger.info(
            "--- Exiting response function due to low STT confidence/word count ---",
        )
        return

    # --- Proceed if STT successful and confidence check passed ---
    # Create user message with metadata
    user_metadata = ChatMessageMetadata(
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        stt_details=stt_metadata_details or None,  # Add STT details if available
    )
    user_message = ChatMessage(role="user", content=prompt, metadata=user_metadata)

    # Add to the *copy* and yield update (convert to dict for frontend compatibility if needed)
    current_chatbot.append(user_message.model_dump())  # Store as dict in the list
    yield AdditionalOutputs({"type": "chatbot_update", "message": user_message.model_dump()})

    # Update messages list (for LLM) based on the modified copy
    messages.append({"role": user_message.role, "content": user_message.content})

    # Save history after adding user message (save_chat_history expects list of dicts)
    save_chat_history(current_chatbot)

    # --- Streaming Chat Completion & Concurrent TTS ---
    llm_response_stream = None
    full_response_text = ""
    sentence_buffer = ""  # Buffer for accumulating raw text including tags
    last_tts_processed_pos = 0  # Tracks the index in the *cleaned* buffer processed for TTS
    final_usage_info = None
    llm_error_occurred = False
    first_chunk_yielded = False  # Track if we yielded the first chunk for UI
    response_completed_normally = False  # Track normal completion
    total_tts_chars = 0  # Initialize TTS character counter
    tts_audio_file_paths: list[str] = []  # List to store FILENAMES of generated TTS audio files for this response

    try:
        # Signal waiting for LLM
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "llm_waiting",
                "message": "Waiting for AI...",
            },
        )
        llm_start_time = time.time()
        logger.info(
            f"Sending prompt to LLM ({settings.current_llm_model}) for streaming...",
        )
        llm_args_dict = {
            "model": settings.current_llm_model,
            "messages": messages,  # Use the history including the user prompt
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },  # Request usage data in the stream
            **({"api_base": settings.llm_api_base} if settings.use_llm_proxy else {}),
        }
        if settings.llm_api_key:  # This is for classic backend LLM
            llm_args_dict["api_key"] = settings.llm_api_key

        llm_response_stream = await litellm.acompletion(**llm_args_dict)

        async for chunk in llm_response_stream:
            logger.debug(f"CHUNK: {chunk}")
            # Check for content delta first
            delta_content = None
            if chunk.choices and chunk.choices[0].delta:
                delta_content = chunk.choices[0].delta.content

            if delta_content:
                # Yield text chunk immediately for UI update
                yield AdditionalOutputs(
                    {"type": "text_chunk_update", "content": delta_content},
                )
                first_chunk_yielded = True

                sentence_buffer += delta_content
                full_response_text += delta_content

            # Check for usage information in the chunk (often in the *last* chunk)
            # LiteLLM might attach it differently depending on the provider.
            # Let's check more robustly.
            chunk_usage = getattr(chunk, "usage", None) or getattr(
                chunk, "_usage", None,
            )  # Check common attributes
            if chunk_usage:
                # If usage is already a dict, use it, otherwise try .dict()
                if isinstance(chunk_usage, dict):
                    final_usage_info = chunk_usage
                elif hasattr(chunk_usage, "dict"):
                    final_usage_info = chunk_usage.dict()
                else:
                    final_usage_info = vars(chunk_usage)  # Fallback to vars()

                logger.info(
                    f"Captured usage info from LLM chunk: {final_usage_info}",
                )

            # --- Process buffer for TTS ---
            # Clean the *entire* current buffer first, then process new parts ending in newline
            if delta_content:  # Only process if there was new content
                buffer_cleaned_for_tts = re.sub(r"<think>.*?</think>", "", sentence_buffer, flags=re.DOTALL)
                new_cleaned_text = buffer_cleaned_for_tts[last_tts_processed_pos:]

                # Find the last newline in the newly added cleaned text
                split_pos = new_cleaned_text.rfind("\n")

                if split_pos != -1:
                    # Extract the chunk ready for TTS (up to and including the last newline)
                    tts_ready_chunk = new_cleaned_text[:split_pos + 1]

                    # Split this chunk into sentences
                    sentences_for_tts = tts_ready_chunk.split("\n")

                    for sentence_for_tts in sentences_for_tts:
                        sentence_for_tts = sentence_for_tts.strip()
                        if sentence_for_tts:
                            # Count characters, log, call generate_tts_for_sentence, handle audio path...
                            total_tts_chars += len(sentence_for_tts)
                            logger.debug(f"Generating TTS for sentence (cleaned): '{sentence_for_tts[:50]}...' ({len(sentence_for_tts)} chars)")
                            audio_file_path: str | None = await generate_tts_for_sentence(
                                sentence_for_tts,
                                settings.tts_client,  # Classic backend TTS client
                                settings.tts_model_arg,
                                settings.current_tts_voice,
                                settings.current_tts_speed,
                                settings.tts_acronym_preserve_set,
                                settings.tts_audio_dir,
                            )
                            if audio_file_path:
                                tts_audio_file_paths.append(audio_file_path)
                                full_audio_file_path = settings.tts_audio_dir / audio_file_path
                                logger.debug(f"TTS audio saved to: {full_audio_file_path}")
                                try:
                                    audio_segment = await asyncio.to_thread(
                                        AudioSegment.from_file, full_audio_file_path, format="mp3",
                                    )
                                    sample_rate = audio_segment.frame_rate
                                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
                                    logger.debug(f"Yielding audio chunk from file '{audio_file_path}' for sentence: '{sentence_for_tts[:50]}...'")
                                    yield (sample_rate, samples)
                                except Exception as read_e:
                                    logger.error(f"Failed to read/decode TTS audio file {full_audio_file_path}: {read_e}")
                            else:
                                logger.warning(f"TTS failed for sentence, skipping audio yield and file save: '{sentence_for_tts[:50]}...'")

                    # Update the position marker for the cleaned buffer
                    last_tts_processed_pos += len(tts_ready_chunk)

        # After the loop, process any remaining text in the cleaned buffer
        buffer_cleaned_for_tts = re.sub(r"<think>.*?</think>", "", sentence_buffer, flags=re.DOTALL)
        remaining_cleaned_text = buffer_cleaned_for_tts[last_tts_processed_pos:].strip()

        if remaining_cleaned_text:
            # Process the final remaining part for TTS
            total_tts_chars += len(remaining_cleaned_text)
            logger.debug(f"Generating TTS for remaining buffer (cleaned): '{remaining_cleaned_text[:50]}...' ({len(remaining_cleaned_text)} chars)")
            audio_file_path: str | None = await generate_tts_for_sentence(
                remaining_cleaned_text,
                settings.tts_client,  # Classic backend TTS client
                settings.tts_model_arg,
                settings.current_tts_voice,
                settings.current_tts_speed,
                settings.tts_acronym_preserve_set,
                settings.tts_audio_dir,
            )
            if audio_file_path:
                tts_audio_file_paths.append(audio_file_path)
                full_audio_file_path = settings.tts_audio_dir / audio_file_path
                logger.debug(f"TTS audio saved to: {full_audio_file_path}")
                try:
                    audio_segment = await asyncio.to_thread(
                        AudioSegment.from_file, full_audio_file_path, format="mp3",
                    )
                    sample_rate = audio_segment.frame_rate
                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
                    logger.debug(f"Yielding audio chunk from file '{audio_file_path}' for remaining buffer: '{remaining_cleaned_text[:50]}...'")
                    yield (sample_rate, samples)
                except Exception as read_e:
                    logger.error(f"Failed to read/decode TTS audio file {full_audio_file_path}: {read_e}")
            else:
                logger.warning(f"TTS failed for remaining buffer, skipping audio yield and file save: '{remaining_cleaned_text[:50]}...'")

        llm_end_time = time.time()
        logger.info(
            f"LLM streaming finished ({llm_end_time - llm_start_time:.2f}s). Full response length: {len(full_response_text)}",
        )
        logger.info(
            f"Total characters sent to TTS: {total_tts_chars}",
        )  # Log total TTS chars

        # --- Final Updates (After LLM stream and TTS generation/yielding) ---
        response_completed_normally = (
            not llm_error_occurred
        )  # Mark normal completion if no LLM error occurred

        # 1. Cost Calculation (LLM and TTS)
        cost_result = {}  # Initialize cost result dict
        tts_cost = 0.0  # Initialize TTS cost

        # Calculate TTS cost if applicable (Classic backend)
        if settings.is_openai_tts and total_tts_chars > 0:
            tts_model_used = settings.tts_model_arg
            if tts_model_used in OPENAI_TTS_PRICING:
                price_per_million_chars = OPENAI_TTS_PRICING[tts_model_used]
                tts_cost = (total_tts_chars / 1_000_000) * price_per_million_chars
                logger.info(
                    f"Calculated OpenAI TTS cost for {total_tts_chars} chars ({tts_model_used}): ${tts_cost:.6f}",
                )
            else:
                logger.warning(
                    f"Cannot calculate TTS cost: Pricing unknown for model '{tts_model_used}'.",
                )
        elif total_tts_chars > 0:
            logger.info(
                f"TTS cost calculation skipped (not using OpenAI TTS or 0 chars). Total chars: {total_tts_chars}",
            )

        cost_result["tts_cost"] = tts_cost  # Add TTS cost (even if 0) to the result

        # Calculate LLM cost (if usage info available - Classic backend)
        if final_usage_info:
            llm_cost_result = calculate_llm_cost(
                settings.current_llm_model, final_usage_info, settings.model_cost_data,
            )
            # Merge LLM cost results into the main cost_result dict
            cost_result.update(llm_cost_result)
            logger.info("LLM cost calculation successful.")
        elif not llm_error_occurred:
            logger.warning(
                "No final usage information received from LLM stream, cannot calculate LLM cost accurately.",
            )
            cost_result["error"] = "LLM usage info missing"
            cost_result["model"] = settings.current_llm_model
            # Ensure LLM cost fields are present but potentially zero or marked
            cost_result.setdefault("input_cost", 0.0)
            cost_result.setdefault("output_cost", 0.0)
            cost_result.setdefault(
                "total_cost", 0.0,
            )

        # Yield combined cost update
        logger.info(f"Yielding combined cost update: {cost_result}")
        yield AdditionalOutputs({"type": "cost_update", "data": cost_result})
        logger.info("Cost update yielded.")

        # 2. Add Full Assistant Text Response to History (to the copy) with Metadata
        assistant_message_obj = None
        if not llm_error_occurred and full_response_text:
            # Create assistant message metadata
            assistant_metadata = ChatMessageMetadata(
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                llm_model=settings.current_llm_model,
                usage=final_usage_info,
                cost=cost_result,
                tts_audio_file_paths=tts_audio_file_paths or None,
            )
            # Create the full message object
            assistant_message_obj = ChatMessage(
                role="assistant",
                content=full_response_text,
                metadata=assistant_metadata,
            )
            assistant_message_dict = assistant_message_obj.model_dump()

            if not current_chatbot or current_chatbot[-1] != assistant_message_dict:
                current_chatbot.append(assistant_message_dict)
                logger.info(
                    "Full assistant response (with metadata) added to chatbot history copy for next turn.",
                )
                # Save history after adding assistant message
                save_chat_history(current_chatbot)
            else:
                logger.info(
                    "Full assistant response already present in history, skipping append.",
                )
        elif not llm_error_occurred:
            logger.warning(
                "LLM stream completed but produced no text content. History not updated.",
            )

        # 3. Yield Final Chatbot State Update (if response completed normally)
        if response_completed_normally:
            logger.info("Yielding final chatbot state update...")
            yield AdditionalOutputs(
                {
                    "type": "final_chatbot_state",
                    "history": current_chatbot,
                },
            )
            logger.info("Final chatbot state update yielded.")

        # 4. Yield Final Status Update (always, should be the last yield in the success path)
        final_status_message = (
            "Ready" if response_completed_normally else "Ready (after error)"
        )
        logger.info(f"Yielding final status update ({final_status_message})...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": final_status_message,
            },
        )
        logger.info("Final status update yielded.")

    except AuthenticationError as e:
        logger.error(f"OpenAI Authentication Error during classic backend processing: {e}")
        response_completed_normally = False
        llm_error_occurred = True

        error_content = f"\n[Authentication Error: Check your API key ({e})]"
        if not first_chunk_yielded:
            llm_error_msg = {"role": "assistant", "content": error_content.strip()}
            yield AdditionalOutputs(
                {"type": "chatbot_update", "message": llm_error_msg},
            )
        else:
            yield AdditionalOutputs(
                {"type": "text_chunk_update", "content": error_content},
            )

        logger.warning("Yielding final chatbot state (after auth exception)...")
        yield AdditionalOutputs(
            {"type": "final_chatbot_state", "history": current_chatbot},
        )
        logger.warning("Final chatbot state (after auth exception) yielded.")

        logger.warning("Yielding final status update (idle, after auth exception)...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (Auth Error)",
            },
        )
        logger.warning("Final status update (idle, after auth exception) yielded.")

    except Exception as e:
        logger.error(
            f"Error during LLM streaming or TTS processing (classic backend): {type(e).__name__} - {e}", exc_info=True,
        )
        response_completed_normally = False
        llm_error_occurred = True

        error_content = f"\n[LLM/TTS Error: {type(e).__name__}]"
        if not first_chunk_yielded:
            llm_error_msg = {"role": "assistant", "content": error_content.strip()}
            yield AdditionalOutputs(
                {"type": "chatbot_update", "message": llm_error_msg},
            )
        else:
            yield AdditionalOutputs(
                {"type": "text_chunk_update", "content": error_content},
            )

        logger.warning("Yielding final chatbot state (after exception)...")
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,
            },
        )
        logger.warning("Final chatbot state (after exception) yielded.")

        logger.warning("Yielding final status update (idle, after exception)...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (after error)",
            },
        )
        logger.warning("Final status update (idle, after exception) yielded.")

    logger.info(
        f"--- Response function generator finished (Completed normally: {response_completed_normally}) ---",
    )

# --- Pydantic Models ---
class ChatMessageMetadata(BaseModel):
    timestamp: str | None = None
    llm_model: str | None = None
    usage: dict[str, Any] | None = None 
    cost: dict[str, Any] | None = None
    stt_details: dict[str, Any] | None = None
    tts_audio_file_paths: list[str] | None = None 
    output_audio_duration_seconds: float | None = None 
    raw_openai_usage_events: list[dict[str, Any]] | None = None 

class ChatMessage(BaseModel):
    role: str
    content: str
    metadata: ChatMessageMetadata | None = None

# --- FastAPI Setup ---
class InputData(BaseModel):
    webrtc_id: str
    chatbot: list[ChatMessage]

# --- Gemini Realtime Handler (Refactored for new API) ---
class GeminiRealtimeHandler(AsyncStreamHandler):
    def __init__(self, app_settings: "AppSettings") -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_REALTIME_OUTPUT_SAMPLE_RATE,
            input_sample_rate=GEMINI_REALTIME_INPUT_SAMPLE_RATE,
        )
        self.settings = app_settings
        self.client: genai.Client | None = None
        self.session: genai.live.AsyncLiveConnectSession | None = None

        self._outgoing_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._audio_sender_task: asyncio.Task | None = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.current_stt_language_code = self.settings.current_stt_language
        self.current_gemini_voice = self.settings.current_gemini_voice


        self._current_input_chars: int = 0
        self._current_output_chars: int = 0
        self._current_input_transcript_parts: list[str] = []
        self._current_output_text_parts: list[str] = []
        self._current_input_audio_duration_this_turn: float = 0.0
        self._current_output_audio_duration_this_turn: float = 0.0
        self._processing_lock = asyncio.Lock()
        self._last_seen_usage_metadata: Any | None = None

    def copy(self):
        return GeminiRealtimeHandler(self.settings)

    def _reset_turn_usage_state(self) -> None:
        self._current_input_chars = 0
        self._current_output_chars = 0
        self._current_input_transcript_parts = []
        self._current_output_text_parts = []
        self._current_input_audio_duration_this_turn = 0.0
        self._current_output_audio_duration_this_turn = 0.0
        self._last_seen_usage_metadata = None
        logger.debug("GeminiRealtimeHandler: Turn usage state reset.")

    async def handle_tool_call(self, tool_calls: list[ToolCall]) -> None:
        logger.debug(f"GeminiRealtimeHandler: Handling tool calls: {tool_calls}")
        global form_data
        function_responses = []

        for tool_call_obj in tool_calls:
            call_id = tool_call_obj.id
            func_call_data = tool_call_obj.function_call
            name = func_call_data.name
            args = func_call_data.args
            response_data = None

            if name == "atualizar_campo_formulario":
                field_key = args.get("chave_campo")
                new_value = args.get("novo_valor")
                if (field_key and new_value is not None and
                    "content" in form_data and
                    field_key in form_data["content"]):
                    form_data["content"][field_key]["value"] = new_value
                    logger.info(f"Campo {field_key} atualizado para: {new_value} via tool call.")
                    await self.output_queue.put(AdditionalOutputs({
                        "type": "update_field_from_ai",
                        "payload": {"key": field_key, "value": new_value},
                    }))
                    response_data = {"status": "success", "field": field_key, "updated_value": new_value}
                else:
                    logger.warning(f"Chave de campo inválida ou dados de formulário ausentes para atualizar: {field_key}")
                    response_data = {"status": "error", "message": f"Invalid field key or form data missing for {field_key}"}
            
            elif name == "obter_descricoes_campos":
                descriptions = {}
                if "content" in form_data:
                    for key, field_desc in form_data["content"].items():
                        descriptions[key] = {
                            "nome_chave": field_desc.get("nome_chave", ""),
                            "descricao": field_desc.get("descricao", ""),
                            "text_precedente": field_desc.get("text_precedente", ""),
                            "value": field_desc.get("value", ""),
                        }
                logger.info(f"Fornecendo descrições de campos para IA: {descriptions}")
                response_data = {"field_descriptions": descriptions}
            else:
                logger.warning(f"Unknown tool call name received: {name}")
                response_data = {"status": "error", "message": f"Unknown tool: {name}"}

            if call_id and response_data is not None:
                function_responses.append(
                    genai.types.FunctionResponse(id=call_id, name=name, response=response_data)
                )
        
        if function_responses and self.session:
            logger.debug(f"Sending tool responses: {function_responses}")
            try:
                await self.session.send_tool_response(function_responses=function_responses)
            except Exception as e:
                logger.error(f"Error sending tool response: {e}", exc_info=True)
        elif not self.session:
            logger.warning("Cannot send tool response: session is not active.")

    async def _audio_sender_task(self):
        logger.info("GeminiRealtimeHandler: Audio sender task started.")
        try:
            while True:
                audio_bytes = await self._outgoing_audio_queue.get()
                if audio_bytes is None: 
                    logger.info("GeminiRealtimeHandler: Audio sender task received shutdown signal.")
                    break
                
                if self.session:
                    try:
                        part = genai.types.Part(inline_data=genai.types.Blob(mime_type='audio/pcm', data=audio_bytes))
                        # Send as a single part Content, role is implicit for client_content
                        await self.session.send_client_content(content=part) 
                        logger.debug(f"Sent audio chunk of {len(audio_bytes)} bytes via send_client_content.")
                    except Exception as e:
                        logger.error(f"Error sending audio chunk via send_client_content: {e}", exc_info=True)
                self._outgoing_audio_queue.task_done()
        except asyncio.CancelledError:
            logger.info("GeminiRealtimeHandler: Audio sender task cancelled.")
        except Exception as e:
            logger.error(f"GeminiRealtimeHandler: Audio sender task encountered an error: {e}", exc_info=True)
        finally:
            logger.info("GeminiRealtimeHandler: Audio sender task finished.")

    async def start_up(self) -> None:
        logger.info("GeminiRealtimeHandler: Starting up and connecting to Gemini...")
        self._outgoing_audio_queue = asyncio.Queue()

        if not self.settings.gemini_api_key:
            logger.error("GeminiRealtimeHandler: Gemini API Key not configured. Cannot connect.")
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": "Gemini API Key missing."}))
            await self.output_queue.put(None) 
            return


        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        
        logger.info(f"GeminiRealtimeHandler: Initializing with STT language: {self.current_stt_language_code or 'auto-detect by API'}, Voice: {self.current_gemini_voice}")
        processed_language_code_for_api = self.current_stt_language_code
        if processed_language_code_for_api and len(processed_language_code_for_api) == 2 and "-" not in processed_language_code_for_api:
            if processed_language_code_for_api.lower() != "pt-br":

                 processed_language_code_for_api = f"{processed_language_code_for_api}-{processed_language_code_for_api.upper()}"
                 logger.info(f"GeminiRealtimeHandler: Formatted 2-letter language code to: {processed_language_code_for_api} for API.")
            else:
                logger.info(f"GeminiRealtimeHandler: Using language code '{processed_language_code_for_api}' as is for API (e.g. pt-BR).")


        speech_config_params: dict[str, Any] = {
            "voice_config": genai.types.VoiceConfig(
                prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(voice_name=self.current_gemini_voice)
            )
        }
        if processed_language_code_for_api:
            speech_config_params["language_code"] = processed_language_code_for_api
        

        function_declarations = [
            {
                "name": "atualizar_campo_formulario", "description": "Atualiza o valor de um campo específico no formulário",
                "parameters": {"type": "object", "properties": {"chave_campo": {"type": "string"}, "novo_valor": {"type": "string"}}, "required": ["chave_campo", "novo_valor"]},
            },
            {
                "name": "obter_descricoes_campos", "description": "Obtém as descrições de todos os campos do formulário",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

        live_connect_config_args: dict[str, Any] = {

            "response_modalities": self.settings.gemini_response_modalities,
            "speech_config": genai.types.SpeechConfig(**speech_config_params),
            "context_window_compression": genai.types.ContextWindowCompressionConfig(
                sliding_window=genai.types.SlidingWindow(),
                trigger_tokens=self.settings.gemini_context_window_compression_threshold,
            ),
            "input_audio_transcription": {}, 
            "output_audio_transcription": {},
            "tools": [{"function_declarations": function_declarations}]
        }

        if self.settings.system_message and self.settings.system_message.strip():
            system_instruction_content = genai.types.Content(parts=[genai.types.Part(text=self.settings.system_message.strip())])
            live_connect_config_args["system_instruction"] = system_instruction_content
        
        live_connect_config = genai.types.LiveConnectConfig(**live_connect_config_args)


        try:
            self._reset_turn_usage_state()
            selected_model = self.settings.current_llm_model or self.settings.gemini_model_arg
            logger.info(f"GeminiRealtimeHandler: Attempting to connect with model {selected_model}, voice {self.current_gemini_voice or 'default'}.")

            async with self.client.aio.live.connect(model=selected_model, config=live_connect_config) as session:
                self.session = session
                logger.info(f"GeminiRealtimeHandler: Connection established.")
                self._reset_turn_usage_state()
                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "stt_processing", "message": "Listening..."}))

                self._audio_sender_task = asyncio.create_task(self._audio_sender_task())

                async for event in self.session.receive():
                    async with self._processing_lock:
                        logger.debug(f"GeminiRealtime Event received: Type: {type(event)}")
                        # logger.debug(f"Full Event: {event}") # Could be too verbose

                        if hasattr(event, "usage_metadata") and event.usage_metadata is not None:
                            self._last_seen_usage_metadata = event.usage_metadata
                        
                        input_transcription = getattr(event, "input_transcription", None)
                        if input_transcription and input_transcription.text:
                            transcript = input_transcription.text
                            is_final = getattr(input_transcription, "is_final", True)
                            logger.debug(f"Gemini STT: '{transcript}' (Final: {is_final})")
                            self._current_input_transcript_parts.append(transcript)
                            if is_final:
                                full_transcript = "".join(self._current_input_transcript_parts)
                                self._current_input_chars = len(full_transcript)
                                user_message = ChatMessage(
                                    role="user", content=full_transcript,
                                    metadata=ChatMessageMetadata(timestamp=datetime.datetime.now(datetime.UTC).isoformat())
                                )
                                await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": user_message.model_dump()}))
                                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "llm_waiting", "message": "AI Responding..."}))
                        
                        model_turn = getattr(event, "model_turn", None)
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                if part.text:
                                    self._current_output_text_parts.append(part.text)
                                
                                inline_data_obj = getattr(part, "inline_data", None)
                                if inline_data_obj and inline_data_obj.data and "audio" in inline_data_obj.mime_type:
                                    output_audio_bytes_chunk = len(inline_data_obj.data)
                                    duration_chunk_output_seconds = output_audio_bytes_chunk / (GEMINI_REALTIME_OUTPUT_SAMPLE_RATE * 2)
                                    self._current_output_audio_duration_this_turn += duration_chunk_output_seconds
                                    audio_data_np = np.frombuffer(inline_data_obj.data, dtype=np.int16)
                                    if audio_data_np.ndim == 1: audio_data_np = audio_data_np.reshape(1, -1)
                                    await self.output_queue.put((GEMINI_REALTIME_OUTPUT_SAMPLE_RATE, audio_data_np))

                        tool_calls = getattr(event, "tool_calls", None)
                        if tool_calls:
                            await self.handle_tool_call(tool_calls)
                        
                        is_turn_ending_event = False
                        end_of_turn_reason = ""
                        if hasattr(event, "speech_processing_event") and event.speech_processing_event.event_type == "END_OF_SINGLE_UTTERANCE":
                            is_turn_ending_event = True
                            end_of_turn_reason = "END_OF_SINGLE_UTTERANCE"
                        elif hasattr(event, "turn_complete") and event.turn_complete:
                             if self._current_input_transcript_parts or self._current_output_text_parts or self._last_seen_usage_metadata:
                                is_turn_ending_event = True
                                end_of_turn_reason = "event.turn_complete"
                        
                        if is_turn_ending_event:
                            logger.info(f"GeminiRealtime: Processing end of turn triggered by: {end_of_turn_reason}.")
                            full_input_transcript = "".join(self._current_input_transcript_parts)
                            if not self._current_input_chars and full_input_transcript:
                                self._current_input_chars = len(full_input_transcript)
                            full_output_text = "".join(self._current_output_text_parts)
                            self._current_output_chars = len(full_output_text)
                            
                            final_usage_metadata_for_turn = getattr(event, "usage_metadata", self._last_seen_usage_metadata)
                            api_prompt_audio_tokens, api_prompt_text_tokens, api_response_audio_tokens = 0,0,0 
                            if final_usage_metadata_for_turn:
                                logger.info(f"GeminiRealtime: Using usage_metadata for cost calculation. Details: {final_usage_metadata_for_turn}")
                                prompt_details = getattr(final_usage_metadata_for_turn, "prompt_tokens_details", [])
                                if not prompt_details:
                                    top_level_prompt_tokens = getattr(final_usage_metadata_for_turn, "prompt_token_count", 0)
                                    if top_level_prompt_tokens > 0: api_prompt_audio_tokens = top_level_prompt_tokens
                                else:
                                    for item in prompt_details:
                                        modality = item.modality.name.upper()
                                        token_count = item.token_count
                                        if modality == "AUDIO": api_prompt_audio_tokens += token_count
                                        elif modality == "TEXT": api_prompt_text_tokens += token_count
                                response_details = getattr(final_usage_metadata_for_turn, "response_tokens_details", [])
                                if not response_details:
                                     top_level_response_tokens = getattr(final_usage_metadata_for_turn, "response_token_count", 0)
                                     if top_level_response_tokens > 0: api_response_audio_tokens = top_level_response_tokens
                                else:
                                    for item in response_details:
                                        modality = item.modality.name.upper()
                                        token_count = item.token_count
                                        if modality == "AUDIO": api_response_audio_tokens += token_count
                            else:
                                logger.warning("GeminiRealtime: No usage_metadata available for cost calculation.")
                            
                            prompt_audio_token_cost, prompt_text_token_cost, response_audio_token_cost = 0.0, 0.0, 0.0
                            if GEMINI_LIVE_PRICING:
                                price_input_audio_per_mil = GEMINI_LIVE_PRICING.get("input_audio_tokens", 0.0)
                                price_input_text_per_mil = GEMINI_LIVE_PRICING.get("input_text_tokens", 0.0)
                                price_output_audio_per_mil = GEMINI_LIVE_PRICING.get("output_audio_tokens", 0.0)
                                prompt_audio_token_cost = (api_prompt_audio_tokens / 1_000_000) * price_input_audio_per_mil
                                prompt_text_token_cost = (api_prompt_text_tokens / 1_000_000) * price_input_text_per_mil
                                response_audio_token_cost = (api_response_audio_tokens / 1_000_000) * price_output_audio_per_mil
                            total_gemini_cost = prompt_audio_token_cost + prompt_text_token_cost + response_audio_token_cost
                            cost_data = {
                                "model": selected_model, "prompt_audio_tokens": api_prompt_audio_tokens,
                                "prompt_text_tokens": api_prompt_text_tokens, "response_audio_tokens": api_response_audio_tokens,
                                "prompt_audio_cost": prompt_audio_token_cost, "prompt_text_cost": prompt_text_token_cost,
                                "response_audio_cost": response_audio_token_cost, "total_cost": total_gemini_cost,
                                "input_audio_duration_seconds": round(self._current_input_audio_duration_this_turn, 3),
                                "output_audio_duration_seconds": round(self._current_output_audio_duration_this_turn, 3),
                                "input_chars": self._current_input_chars, "output_chars": self._current_output_chars,
                                "note": "Costs are based on API-provided token counts per modality from usage_metadata.",
                            }
                            await self.output_queue.put(AdditionalOutputs({"type": "cost_update", "data": cost_data}))
                            assistant_metadata = ChatMessageMetadata(
                                timestamp=datetime.datetime.now(datetime.UTC).isoformat(), llm_model=selected_model, cost=cost_data, 
                                usage={
                                    "prompt_audio_tokens": api_prompt_audio_tokens, "prompt_text_tokens": api_prompt_text_tokens,
                                    "response_audio_tokens": api_response_audio_tokens,
                                    "input_audio_duration_seconds": round(self._current_input_audio_duration_this_turn, 3),
                                    "output_audio_duration_seconds": round(self._current_output_audio_duration_this_turn, 3),
                                    "input_chars": self._current_input_chars, "output_chars": self._current_output_chars,
                                }
                            )
                            assistant_message = ChatMessage(role="assistant", content=full_output_text, metadata=assistant_metadata)
                            await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": assistant_message.model_dump()}))
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "idle", "message": "Ready"}))
                            self._reset_turn_usage_state()

                        error_obj = getattr(event, "error", None)
                        if error_obj:
                            logger.error(f"GeminiRealtime API Error: Code {error_obj.code}, Message: {error_obj.message}")
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Gemini Error: {error_obj.message}"}))
                            error_chat_message = ChatMessage(role="assistant", content=f"[Gemini Error: {error_obj.message}]")
                            await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": error_chat_message.model_dump()}))
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "idle", "message": "Ready (after error)"}))
                            self._reset_turn_usage_state()
            
        except Exception as e:
            logger.error(f"GeminiRealtimeHandler: Connection failed or error during session: {e}", exc_info=True)
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Connection Error: {e!s}"}))
        finally:
            logger.info("GeminiRealtimeHandler: start_up processing loop finished.")
            if self._audio_sender_task: 
                await self._outgoing_audio_queue.put(None) 
                try:
                    await asyncio.wait_for(self._audio_sender_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("GeminiRealtimeHandler: Audio sender task did not finish in time during shutdown.")
                except asyncio.CancelledError:
                     logger.info("GeminiRealtimeHandler: Audio sender task was cancelled during shutdown.")
                self._audio_sender_task = None
            
            self.session = None
            await self.output_queue.put(None) 

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.session and not self._audio_sender_task: 
            logger.warning("GeminiRealtimeHandler: Session not active and sender task not running, dropping audio frame.")
            return

        _, array = frame
        if array.ndim > 1: array = array.squeeze()
        if array.dtype != np.int16: array = array.astype(np.int16)
        audio_bytes = array.tobytes()
        
        if hasattr(self, '_outgoing_audio_queue') and self._outgoing_audio_queue is not None:
            try:
                self._outgoing_audio_queue.put_nowait(audio_bytes)
            except asyncio.QueueFull:
                logger.warning("GeminiRealtimeHandler: Outgoing audio queue is full. Dropping frame.")
            except Exception as e: 
                logger.error(f"GeminiRealtimeHandler: Error putting audio to outgoing_audio_queue: {e}")
        else:
            logger.warning("GeminiRealtimeHandler: _outgoing_audio_queue not initialized or already closed. Cannot send audio.")


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        logger.info("GeminiRealtimeHandler: Shutting down...")
        if self._audio_sender_task and not self._audio_sender_task.done():
            logger.info("GeminiRealtimeHandler: Signaling audio sender task to stop.")
            if hasattr(self, '_outgoing_audio_queue') and self._outgoing_audio_queue is not None:
                 await self._outgoing_audio_queue.put(None) 
            try:
                await asyncio.wait_for(self._audio_sender_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("GeminiRealtimeHandler: Audio sender task did not stop in time during shutdown.")
                self._audio_sender_task.cancel() 
            except asyncio.CancelledError:
                logger.info("GeminiRealtimeHandler: Audio sender task was already cancelled.")
            except Exception as e:
                logger.error(f"GeminiRealtimeHandler: Error during audio sender task shutdown: {e}")
        self._audio_sender_task = None
        
        self.session = None 

        if hasattr(self, '_outgoing_audio_queue') and self._outgoing_audio_queue is not None:
            while not self._outgoing_audio_queue.empty():
                try:
                    self._outgoing_audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.debug("GeminiRealtimeHandler: Outgoing audio queue drained.")

        self.clear_output_queue()
        if self.output_queue: 
            await self.output_queue.put(None)
        logger.info("GeminiRealtimeHandler: Shutdown complete.")

    def clear_output_queue(self) -> None:  
        if hasattr(self, 'output_queue') and self.output_queue is not None:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.debug("GeminiRealtimeHandler: Output queue cleared.")

# --- OpenAI Realtime Handler ---
# ... (OpenAIRealtimeHandler class - unchanged) ...
# --- Endpoint Definitions ---
# ... (register_endpoints function - unchanged) ...
# --- Pywebview API Class ---
# ... (Api class - unchanged) ...
# --- Heartbeat Monitoring Thread ---


def monitor_heartbeat_thread() -> None:
    global last_heartbeat_time, uvicorn_server, pywebview_window, shutdown_event
    logger.info("Heartbeat monitor thread started.")
    initial_wait_done = False

    while not shutdown_event.is_set():
        if last_heartbeat_time is None:
            if not initial_wait_done:
                logger.info(
                    f"Waiting for the first heartbeat (timeout check in {heartbeat_timeout * 2}s)...",
                )
                shutdown_event.wait(heartbeat_timeout * 2)
                initial_wait_done = True
                if shutdown_event.is_set():
                    break
                continue
            logger.debug("Still waiting for first heartbeat...")
            shutdown_event.wait(5)
            if shutdown_event.is_set():
                break
            continue

        time_since_last = (
            datetime.datetime.now(datetime.UTC) - last_heartbeat_time
        )
        logger.debug(
            f"Time since last heartbeat: {time_since_last.total_seconds():.1f}s",
        )

        if time_since_last.total_seconds() > heartbeat_timeout:
            if not settings.disable_heartbeat:
                logger.warning(
                    f"Heartbeat timeout ({heartbeat_timeout}s exceeded). Initiating shutdown.",
                )
                if uvicorn_server:
                    logger.info("Signaling Uvicorn server to stop...")
                    uvicorn_server.should_exit = True
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown.",
                    )

                if pywebview_window:
                    logger.info("Destroying pywebview window...")
                    try:
                        pywebview_window.destroy()
                    except Exception as e:
                        logger.error(
                            f"Error destroying pywebview window from monitor thread: {e}",
                        )
                break  # Break loop to terminate monitor thread after shutdown initiated
            logger.info(
                f"Heartbeat timeout ({heartbeat_timeout}s exceeded), but heartbeat monitoring is disabled. Not shutting down.",
            )
            # Reset last_heartbeat_time to prevent constant logging of this message if client truly disconnected badly
            # This means we'd only log this once, then wait for a new "first" heartbeat.
            last_heartbeat_time = None
            initial_wait_done = False  # Re-trigger initial wait logic
            logger.info("Resetting heartbeat state to wait for a new initial heartbeat.")

        shutdown_event.wait(5)
    logger.info("Heartbeat monitor thread finished.")


@click.command(help="Run a simple voice chat interface using a configurable LLM provider, STT server, and TTS.")
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind the FastAPI server to.",
)
@click.option(
    "--port",
    type=click.INT,
    envvar="APP_PORT",
    default=int(APP_PORT_ENV),
    show_default=True,
    help="Preferred port to run the FastAPI server on. (Env: APP_PORT)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose logging (DEBUG level).",
)
@click.option(
    "--browser",
    is_flag=True,
    default=False,
    show_default=True,
    help="Launch the application in the default web browser instead of a dedicated GUI window.",
)
@click.option(
    "--system-message",
    type=str,
    envvar="SYSTEM_MESSAGE",
    default=SYSTEM_MESSAGE_ENV,
    show_default=True,
    help="System message to prepend to the chat history. (Env: SYSTEM_MESSAGE)",
)
@click.option(
    "--disable-heartbeat",
    is_flag=True,
    envvar="DISABLE_HEARTBEAT",
    default=(DISABLE_HEARTBEAT_ENV.lower() == "true"),  # Convert string "False" to bool False
    show_default=True,
    help="Disable heartbeat timeout check (application will not exit if browser tab is closed without proper shutdown). (Env: DISABLE_HEARTBEAT)",
)
@click.option(
    "--backend",
    type=click.Choice(["classic", "openai", "gemini"], case_sensitive=False),  # Add "gemini"
    default="classic",
    show_default=True,
    help="Backend to use for voice processing. 'classic' uses separate STT/LLM/TTS. 'openai' uses OpenAI's realtime voice API. 'gemini' uses Google's Gemini Live Connect API (Alpha).",
)
@click.option(
    "--openai-realtime-model",
    type=str,
    envvar="OPENAI_REALTIME_MODEL",
    default=OPENAI_REALTIME_MODEL_ENV,
    show_default=True,
    help="OpenAI realtime API model to use (if --backend=openai). (Env: OPENAI_REALTIME_MODEL)",
)
@click.option(
    "--openai-realtime-voice",
    type=str,
    envvar="OPENAI_REALTIME_VOICE",
    default=OPENAI_REALTIME_VOICE_ENV,
    show_default=True,
    help="Default voice for OpenAI realtime backend (if --backend=openai). (Env: OPENAI_REALTIME_VOICE)",
)
@click.option(
    "--openai-api-key",
    type=str,
    envvar="OPENAI_API_KEY",
    default=OPENAI_API_KEY_ENV,
    show_default=True,
    help="API key for OpenAI services (REQUIRED if --backend=openai). (Env: OPENAI_API_KEY)",
)
# Add Gemini CLI options
@click.option(
    "--gemini-model",
    type=str,
    envvar="GEMINI_MODEL",
    default=GEMINI_MODEL_ENV,
    show_default=True,
    help="Gemini model to use (if --backend=gemini). (Env: GEMINI_MODEL)",
)
@click.option(
    "--gemini-voice",
    type=str,
    envvar="GEMINI_VOICE",
    default=GEMINI_VOICE_ENV,
    show_default=True,
    help="Default voice for Gemini backend (if --backend=gemini). (Env: GEMINI_VOICE)",
)
@click.option(
    "--gemini-api-key",
    type=str,
    envvar="GEMINI_API_KEY",
    default=GEMINI_API_KEY_ENV,
    show_default=True,
    help="API key for Google Gemini services (REQUIRED if --backend=gemini). (Env: GEMINI_API_KEY)",
)
@click.option(
    "--gemini-compression-threshold",
    type=click.INT,
    envvar="GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD",
    default=int(GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD_ENV),
    show_default=True,
    help="Context window compression threshold for Gemini backend (if --backend=gemini). (Env: GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD)",
)
@click.option(
    "--default-system-instruction",
    type=str,
    envvar="DEFAULT_SYSTEM_INSTRUCTION",
    default=DEFAULT_SYSTEM_INSTRUCTION_ENV,
    show_default=True,
    help="Default system instruction for Gemini backend if no specific system message is provided. (Env: DEFAULT_SYSTEM_INSTRUCTION)",
)
@click.option(
    "--llm-host",
    type=str,
    envvar="LLM_HOST",
    default=LLM_HOST_ENV,
    show_default=True,
    help="Host address of the LLM proxy server (classic backend, optional). (Env: LLM_HOST)",
)
@click.option(
    "--llm-port",
    type=str,
    envvar="LLM_PORT",
    default=LLM_PORT_ENV,
    show_default=True,
    help="Port of the LLM proxy server (classic backend, optional). (Env: LLM_PORT)",
)
@click.option(
    "--llm-model",
    type=str,
    envvar="LLM_MODEL",
    default=DEFAULT_LLM_MODEL_ENV,
    show_default=True,
    help="Default LLM model to use for classic backend (e.g., 'gpt-4o', 'litellm_proxy/claude-3-opus'). (Env: LLM_MODEL)",
)
@click.option(
    "--llm-api-key",
    type=str,
    envvar="LLM_API_KEY",
    default=LLM_API_KEY_ENV,
    show_default=True,
    help="API key for the LLM provider/proxy (classic backend, optional). (Env: LLM_API_KEY)",
)
@click.option(
    "--stt-host",
    type=str,
    envvar="STT_HOST",
    default=STT_HOST_ENV,
    show_default=True,
    help="Host address of the STT server (classic backend). (Env: STT_HOST)",
)
@click.option(
    "--stt-port",
    type=str,
    envvar="STT_PORT",
    default=STT_PORT_ENV,
    show_default=True,
    help="Port of the STT server (classic backend). (Env: STT_PORT)",
)
@click.option(
    "--stt-model",
    type=str,
    envvar="STT_MODEL",
    default=STT_MODEL_ENV,
    show_default=True,
    help="STT model to use (classic backend). (Env: STT_MODEL)",
)
@click.option(
    "--stt-language",
    type=str,
    envvar="STT_LANGUAGE",
    default=STT_LANGUAGE_ENV,
    show_default=True,
    help="Language code for STT (e.g., 'en', 'fr'). Used by both backends. If unset, Whisper usually auto-detects. (Env: STT_LANGUAGE)",
)
@click.option(
    "--stt-api-key",
    type=str,
    envvar="STT_API_KEY",
    default=STT_API_KEY_ENV,
    show_default=True,
    help="API key for the STT server (classic backend, e.g., for OpenAI STT). (Env: STT_API_KEY)",
)
@click.option(
    "--stt-no-speech-prob-threshold",
    type=click.FLOAT,
    envvar="STT_NO_SPEECH_PROB_THRESHOLD",
    default=float(STT_NO_SPEECH_PROB_THRESHOLD_ENV),
    show_default=True,
    help="STT confidence (classic backend): Reject if no_speech_prob > this. (Env: STT_NO_SPEECH_PROB_THRESHOLD)",
)
@click.option(
    "--stt-avg-logprob-threshold",
    type=click.FLOAT,
    envvar="STT_AVG_LOGPROB_THRESHOLD",
    default=float(STT_AVG_LOGPROB_THRESHOLD_ENV),
    show_default=True,
    help="STT confidence (classic backend): Reject if avg_logprob < this. (Env: STT_AVG_LOGPROB_THRESHOLD)",
)
@click.option(
    "--stt-min-words-threshold",
    type=click.INT,
    envvar="STT_MIN_WORDS_THRESHOLD",
    default=int(STT_MIN_WORDS_THRESHOLD_ENV),
    show_default=True,
    help="STT confidence (classic backend): Reject if word count < this. (Env: STT_MIN_WORDS_THRESHOLD)",
)
@click.option(
    "--tts-host",
    type=str,
    envvar="TTS_HOST",
    default=TTS_HOST_ENV,
    show_default=True,
    help="Host address of the TTS server (classic backend). (Env: TTS_HOST)",
)
@click.option(
    "--tts-port",
    type=str,
    envvar="TTS_PORT",
    default=TTS_PORT_ENV,
    show_default=True,
    help="Port of the TTS server (classic backend). (Env: TTS_PORT)",
)
@click.option(
    "--tts-model",
    type=str,
    envvar="TTS_MODEL",
    default=TTS_MODEL_ENV,
    show_default=True,
    help="TTS model to use (classic backend). (Env: TTS_MODEL)",
)
@click.option(
    "--tts-voice",
    type=str,
    envvar="TTS_VOICE",  # This is for CLASSIC backend TTS voice
    default=DEFAULT_VOICE_TTS_ENV,
    show_default=True,
    help="Default TTS voice to use (classic backend). (Env: TTS_VOICE)",
)
@click.option(
    "--tts-api-key",
    type=str,
    envvar="TTS_API_KEY",
    default=TTS_API_KEY_ENV,
    show_default=True,
    help="API key for the TTS server (classic backend, e.g., for OpenAI TTS). (Env: TTS_API_KEY)",
)
@click.option(
    "--tts-speed",
    type=click.FLOAT,
    envvar="TTS_SPEED",
    default=float(DEFAULT_TTS_SPEED_ENV),
    show_default=True,
    help="Default TTS speed multiplier (classic backend). (Env: TTS_SPEED)",
)
@click.option(
    "--tts-acronym-preserve-list",
    type=str,
    envvar="TTS_ACRONYM_PRESERVE_LIST",
    default=TTS_ACRONYM_PRESERVE_LIST_ENV,
    show_default=True,
    help="Comma-separated list of acronyms to preserve during TTS (classic backend, Kokoro TTS). (Env: TTS_ACRONYM_PRESERVE_LIST)",
)
def main(
    host: str,
    port: int,
    verbose: bool,
    browser: bool,
    system_message: str | None,
    disable_heartbeat: bool,
    backend: str,
    openai_realtime_model: str,
    openai_realtime_voice: str,  # New CLI option for OpenAI backend voice
    openai_api_key: str | None,
    gemini_model: str,             # Add Gemini arg
    gemini_voice: str,             # Add Gemini arg
    gemini_api_key: str | None,  # Add Gemini arg
    gemini_compression_threshold: int,  # Add Gemini compression threshold arg
    default_system_instruction_arg: str,
    llm_host: str | None,
    llm_port: str | None,
    llm_model: str,
    llm_api_key: str | None,
    stt_host: str,
    stt_port: str,
    stt_model: str,
    stt_language: str | None,
    stt_api_key: str | None,
    stt_no_speech_prob_threshold: float,
    stt_avg_logprob_threshold: float,
    stt_min_words_threshold: int,
    tts_host: str,
    tts_port: str,
    tts_model: str,
    tts_voice: str,  # This is for CLASSIC backend TTS voice
    tts_api_key: str | None,
    tts_speed: float,
    tts_acronym_preserve_list: str,
) -> int:
    global uvicorn_server, pywebview_window

    startup_time = datetime.datetime.now()
    startup_timestamp_str_local = startup_time.strftime("%Y%m%d_%H%M%S")

    settings.startup_timestamp_str = startup_timestamp_str_local
    settings.backend = backend
    settings.openai_realtime_model_arg = openai_realtime_model
    settings.openai_realtime_voice_arg = openai_realtime_voice  # Store initial arg
    settings.openai_api_key = openai_api_key
    settings.gemini_model_arg = gemini_model       # Store Gemini arg
    settings.gemini_voice_arg = gemini_voice       # Store Gemini arg
    settings.gemini_api_key = gemini_api_key       # Store Gemini arg
    settings.gemini_context_window_compression_threshold = gemini_compression_threshold  # Store Gemini threshold
    settings.default_system_instruction = default_system_instruction_arg # Store new arg for AppSettings

    settings.preferred_port = port
    settings.host = host
    settings.verbose = verbose
    settings.browser = browser
    settings.system_message = system_message.strip() if system_message is not None else ""
    settings.disable_heartbeat = disable_heartbeat

    # --- Logging Setup (Early) ---
    console_log_level_str = "DEBUG" if settings.verbose else "INFO"
    log_file_path_for_setup: Path | None = None
    log_dir_creation_error_details: str | None = None
    load_form_data()
    try:
        app_name = "SimpleVoiceChat"
        app_author = "Attila"
        log_base_dir_path_str: str | None = None
        try:
            log_base_dir_path_str = platformdirs.user_log_dir(app_name, app_author)
        except Exception:
            try:
                log_base_dir_path_str = platformdirs.user_data_dir(app_name, app_author)
            except Exception as e_data_dir:
                log_dir_creation_error_details = f"Could not find user data directory either ({e_data_dir})."

        if log_base_dir_path_str:
            log_base_dir = Path(log_base_dir_path_str)
            settings.app_log_dir = log_base_dir / "logs"
            settings.app_log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path_for_setup = settings.app_log_dir / f"log_{settings.startup_timestamp_str}.log"
        elif not log_dir_creation_error_details:
            log_dir_creation_error_details = "Failed to determine a valid base directory for logs."

    except Exception as e:
        log_dir_creation_error_details = f"Failed to set up log directory structure: {e}."

    setup_logging(console_log_level_str, log_file_path_for_setup, settings.verbose)
    if log_dir_creation_error_details:
        logger.error(f"Log directory setup failed: {log_dir_creation_error_details} File logging is disabled.")

    logger.info(f"Application Version: {APP_VERSION}")
    logger.info(f"Using backend: {settings.backend}")
    if settings.backend == "gemini":
        logger.warning("The Gemini Live Connect API backend is experimental (uses v1alpha).")

    # --- STT Language (Common to all backends that support it) ---
    settings.stt_language_arg = stt_language
    if settings.stt_language_arg:
        logger.info(f"STT language specified: {settings.stt_language_arg}")
        settings.current_stt_language = settings.stt_language_arg
    else:
        logger.info("No STT language specified (or empty), STT will auto-detect initially.")
        settings.current_stt_language = None # This might be overridden by Gemini specific config below

    # --- Backend Specific Configuration ---
    if settings.backend == "openai":
        logger.info("Configuring for 'openai' backend.")

        settings.available_models = OPENAI_REALTIME_MODELS  # Use the list from config

        if not settings.available_models:
            logger.critical(
                "OPENAI_REALTIME_MODELS list is empty in configuration. "
                "Cannot proceed with OpenAI backend. Exiting.",
            )
            return 1

        # Validate the chosen model (from CLI/env, stored in settings.openai_realtime_model_arg)
        # against the available list.
        chosen_model = settings.openai_realtime_model_arg
        if chosen_model in settings.available_models:
            settings.current_llm_model = chosen_model
        else:
            logger.warning(
                f"OpenAI realtime model '{chosen_model}' (from --openai-realtime-model or env) "
                f"is not in the list of supported models: {settings.available_models}. "
                f"Defaulting to the first available model: '{settings.available_models[0]}'.",
            )
            settings.current_llm_model = settings.available_models[0]

        logger.info(f"OpenAI Realtime Model set to: {settings.current_llm_model}")
        settings.model_cost_data = {}  # Cost for OpenAI realtime is handled differently (per token via OPENAI_REALTIME_PRICING)

        if not settings.openai_api_key:
            logger.critical("OpenAI API Key (--openai-api-key or OPENAI_API_KEY env) is REQUIRED for 'openai' backend. Exiting.")
            return 1
        logger.info("Using dedicated OpenAI API Key for 'openai' backend.")

        # OpenAI Realtime Voice
        settings.available_voices_tts = OPENAI_REALTIME_VOICES  # Populate for UI dropdown
        initial_openai_voice_preference = settings.openai_realtime_voice_arg
        if initial_openai_voice_preference and initial_openai_voice_preference in OPENAI_REALTIME_VOICES:
            settings.current_openai_voice = initial_openai_voice_preference
        elif OPENAI_REALTIME_VOICES:
            if initial_openai_voice_preference:
                 logger.warning(f"OpenAI realtime voice '{initial_openai_voice_preference}' not found. Using first available: {OPENAI_REALTIME_VOICES[0]}.")
            settings.current_openai_voice = OPENAI_REALTIME_VOICES[0]
        else:  # Should not happen if OPENAI_REALTIME_VOICES is populated
            settings.current_openai_voice = initial_openai_voice_preference
            logger.error(f"No OpenAI realtime voices available or specified voice '{settings.current_openai_voice}' is invalid. Voice may not work.")
        logger.info(f"Initial OpenAI realtime voice set to: {settings.current_openai_voice}")

        # Log warnings if classic backend STT/TTS params are provided unnecessarily
        if stt_host != STT_HOST_ENV or stt_port != STT_PORT_ENV or stt_model != STT_MODEL_ENV or stt_api_key is not None:
            logger.warning("STT host/port/model/api-key parameters are ignored when using 'openai' backend (except STT language).")
        if tts_host != TTS_HOST_ENV or tts_port != TTS_PORT_ENV or tts_model != TTS_MODEL_ENV or tts_voice != DEFAULT_VOICE_TTS_ENV or tts_api_key is not None or tts_speed != float(DEFAULT_TTS_SPEED_ENV):
            logger.warning("Classic TTS host/port/model/voice/api-key/speed parameters are ignored when using 'openai' backend.")

        settings.current_tts_speed = 1.0  # Not user-configurable for OpenAI backend via this app

    elif settings.backend == "gemini":
        logger.info("Configuring for 'gemini' backend.")
        settings.available_models = GEMINI_LIVE_MODELS

        if not settings.available_models:
            logger.critical("GEMINI_LIVE_MODELS list is empty. Cannot proceed with Gemini. Exiting.")
            return 1

        # Ensure current_llm_model is set to gemini_model_arg for Gemini backend
        settings.current_llm_model = settings.gemini_model_arg
        if settings.current_llm_model not in settings.available_models:
            logger.warning(
                f"Gemini model '{settings.current_llm_model}' (from --gemini-model or env) "
                f"is not in the list of supported models: {settings.available_models}. "
                f"Defaulting to the first available model: '{settings.available_models[0]}'.",
            )
            settings.current_llm_model = settings.available_models[0]
        logger.info(f"Gemini Live Model set to: {settings.current_llm_model}")
        settings.model_cost_data = {} # Cost handled by GeminiRealtimeHandler

        if not settings.gemini_api_key:
            logger.critical("Gemini API Key (--gemini-api-key or GEMINI_API_KEY env) is REQUIRED for 'gemini' backend. Exiting.")
            return 1
        logger.info("Using dedicated Gemini API Key for 'gemini' backend.")

        # Ensure current_gemini_voice is set from AppSettings.gemini_voice_name
        settings.current_gemini_voice = settings.gemini_voice_name
        if settings.current_gemini_voice not in GEMINI_LIVE_VOICES:
            logger.warning(
                f"Default Gemini voice '{settings.current_gemini_voice}' from AppSettings.gemini_voice_name "
                f"is not in the available list: {GEMINI_LIVE_VOICES}. "
                f"Using the first available voice: {GEMINI_LIVE_VOICES[0] if GEMINI_LIVE_VOICES else 'None'}.",
            )
            if GEMINI_LIVE_VOICES:
                settings.current_gemini_voice = GEMINI_LIVE_VOICES[0]
            else:
                logger.error("No Gemini voices available in GEMINI_LIVE_VOICES. Voice may not work.")
                settings.current_gemini_voice = None # Or handle as critical error
        logger.info(f"Initial Gemini voice set to: {settings.current_gemini_voice}")

        # If system_message (from CLI/env) is empty, use default_system_instruction
        if not settings.system_message or settings.system_message.isspace():
            logger.info("System message is empty or whitespace, using default system instruction for Gemini backend.")
            settings.system_message = settings.default_system_instruction
        logger.info(f"System message for Gemini backend: '{settings.system_message[:100]}...'")

        # If stt_language_arg (from CLI/env) was not provided, use gemini_language_code from AppSettings
        # settings.current_stt_language is already populated by stt_language_arg or None.
        if settings.stt_language_arg is None or not settings.stt_language_arg.strip():
            logger.info(f"No STT language argument provided (or it was empty) for Gemini backend. Using default from AppSettings: {settings.gemini_language_code}")
            settings.current_stt_language = settings.gemini_language_code
        else:
            # If stt_language_arg was provided, settings.current_stt_language already holds its value.
             logger.info(f"Using STT language from --stt-language CLI/env for Gemini backend: {settings.current_stt_language}")
        logger.info(f"Final STT language for Gemini backend: {settings.current_stt_language or 'None (auto-detect by API if not set by gemini_language_code)'}")

        # Log warnings for unused classic backend parameters
        if stt_host != STT_HOST_ENV or stt_port != STT_PORT_ENV or stt_model != STT_MODEL_ENV or stt_api_key is not None:
            logger.warning("Classic STT host/port/model/api-key parameters are ignored when using 'gemini' backend (except STT language via --stt-language).")
        if tts_host != TTS_HOST_ENV or tts_port != TTS_PORT_ENV or tts_model != TTS_MODEL_ENV or tts_voice != DEFAULT_VOICE_TTS_ENV or tts_api_key is not None or tts_speed != float(DEFAULT_TTS_SPEED_ENV):
            logger.warning("Classic TTS host/port/model/voice/api-key/speed parameters are ignored when using 'gemini' backend.")

        settings.current_tts_speed = 1.0 # Not user-configurable for Gemini

    elif settings.backend == "classic":
        logger.info("Configuring for 'classic' backend.")
        # --- LLM Configuration (Classic) ---
        settings.llm_host_arg = llm_host
        settings.llm_port_arg = llm_port
        settings.llm_model_arg = llm_model
        settings.llm_api_key = llm_api_key

        settings.use_llm_proxy = bool(settings.llm_host_arg and settings.llm_port_arg)
        if settings.use_llm_proxy:
            try:
                llm_port_int = int(settings.llm_port_arg)  # type: ignore
                settings.llm_api_base = f"http://{settings.llm_host_arg}:{llm_port_int}/v1"
                logger.info(f"Using LLM proxy at: {settings.llm_api_base}")
                if settings.llm_api_key:
                    logger.info("Using LLM API key for proxy.")
                else:
                    logger.info("No LLM API key provided for proxy (assumed optional).")
            except (ValueError, TypeError):
                logger.error(
                    f"Error: Invalid LLM port specified: '{settings.llm_port_arg}'. Disabling proxy.",
                )
                settings.use_llm_proxy = False
                settings.llm_api_base = None
        else:
            settings.llm_api_base = None
            logger.info("Not using LLM proxy (using default LLM routing).")
            if settings.llm_api_key:
                logger.info("Using LLM API key for direct routing.")
            else:
                logger.info("No LLM API key provided for direct routing (will use LiteLLM's environment config).")

        # --- STT Configuration (Classic) ---
        settings.stt_host_arg = stt_host
        settings.stt_port_arg = stt_port
        settings.stt_model_arg = stt_model
        settings.stt_api_key = stt_api_key
        settings.stt_no_speech_prob_threshold = stt_no_speech_prob_threshold
        settings.stt_avg_logprob_threshold = stt_avg_logprob_threshold
        settings.stt_min_words_threshold = stt_min_words_threshold

        settings.is_openai_stt = settings.stt_host_arg == "api.openai.com"
        if settings.is_openai_stt:
            settings.stt_api_base = "https://api.openai.com/v1"
            logger.info(f"Using OpenAI STT at: {settings.stt_api_base} with model {settings.stt_model_arg}")
            if not settings.stt_api_key:
                logger.critical(
                    "STT_API_KEY (--stt-api-key) is required when using OpenAI STT (stt-host=api.openai.com) with classic backend. Exiting.",
                )
                return 1
            logger.info("Using STT API key for OpenAI STT.")
        else:
            try:
                stt_port_int = int(settings.stt_port_arg)
                scheme = "http"
                settings.stt_api_base = f"{scheme}://{settings.stt_host_arg}:{stt_port_int}/v1"
                logger.info(f"Using Custom STT server at: {settings.stt_api_base} with model {settings.stt_model_arg}")
                if settings.stt_api_key:
                    logger.info("Using STT API key for custom STT server.")
                else:
                    logger.info("No STT API key provided for custom STT server (assumed optional).")
            except (ValueError, TypeError):
                logger.critical(
                    f"Invalid STT port specified for custom server: '{settings.stt_port_arg}'. Cannot connect. Exiting.",
                )
                return 1
        logger.info(
            f"STT Confidence Thresholds: no_speech_prob > {settings.stt_no_speech_prob_threshold}, avg_logprob < {settings.stt_avg_logprob_threshold}, min_words < {settings.stt_min_words_threshold}",
        )

        # --- TTS Configuration (Classic) ---
        settings.tts_host_arg = tts_host
        settings.tts_port_arg = tts_port
        settings.tts_model_arg = tts_model
        settings.tts_voice_arg = tts_voice  # Classic backend TTS voice
        settings.tts_api_key = tts_api_key
        settings.tts_speed_arg = tts_speed
        settings.tts_acronym_preserve_list_arg = tts_acronym_preserve_list

        settings.is_openai_tts = settings.tts_host_arg == "api.openai.com"
        if settings.is_openai_tts:
            settings.tts_base_url = "https://api.openai.com/v1"
            logger.info(f"Using OpenAI TTS at: {settings.tts_base_url} with model {settings.tts_model_arg}")
            if not settings.tts_api_key:
                logger.critical(
                    "TTS_API_KEY (--tts-api-key) is required when using OpenAI TTS (tts-host=api.openai.com) with classic backend. Exiting.",
                )
                return 1
            logger.info("Using TTS API key for OpenAI TTS.")
            if settings.tts_model_arg in OPENAI_TTS_PRICING:
                logger.info(
                    f"OpenAI TTS pricing for '{settings.tts_model_arg}': ${OPENAI_TTS_PRICING[settings.tts_model_arg]:.2f} / 1M chars",
                )
        else:
            try:
                tts_port_int = int(settings.tts_port_arg)
                scheme = "http"
                settings.tts_base_url = f"{scheme}://{settings.tts_host_arg}:{tts_port_int}/v1"
                logger.info(f"Using Custom TTS server at: {settings.tts_base_url} with model {settings.tts_model_arg}")
                if settings.tts_api_key:
                    logger.info("Using TTS API key for custom TTS server.")
                else:
                    logger.info("No TTS API key provided for custom TTS server (assumed optional).")
            except (ValueError, TypeError):
                logger.critical(
                    f"Invalid TTS port specified for custom server: '{settings.tts_port_arg}'. Cannot connect. Exiting.",
                )
                return 1

        settings.tts_acronym_preserve_set = {
            word.strip().upper()
            for word in settings.tts_acronym_preserve_list_arg.split(",")
            if word.strip()
        }
        logger.debug(f"Loaded TTS_ACRONYM_PRESERVE_SET: {settings.tts_acronym_preserve_set}")
        settings.current_tts_speed = settings.tts_speed_arg
        logger.info(f"Initial TTS speed (classic backend): {settings.current_tts_speed:.1f}")

        # --- Initialize Clients (Classic Backend) ---
        logger.info("Initializing clients for 'classic' backend...")
        try:
            settings.stt_client = OpenAI(
                base_url=settings.stt_api_base,
                api_key=settings.stt_api_key,
            )
            logger.info(f"STT client initialized for classic backend (target: {settings.stt_api_base}).")
        except Exception as e:
            logger.critical(f"Failed to initialize STT client for classic backend: {e}. Exiting.")
            return 1

        try:
            settings.tts_client = OpenAI(
                base_url=settings.tts_base_url,
                api_key=settings.tts_api_key,
            )
            logger.info(f"TTS client initialized for classic backend (target: {settings.tts_base_url}).")
        except Exception as e:
            logger.critical(f"Failed to initialize TTS client for classic backend: {e}. Exiting.")
            return 1

        # --- Model & Voice Availability (Classic Backend) ---
        if settings.use_llm_proxy:
            settings.available_models, settings.model_cost_data = get_models_and_costs_from_proxy(
                settings.llm_api_base, settings.llm_api_key,
            )
        else:
            settings.available_models, settings.model_cost_data = get_models_and_costs_from_litellm()

        if not settings.available_models:
            logger.warning("No LLM models found from proxy or litellm. Using fallback.")
            settings.available_models = ["fallback/unknown-model"]

        initial_llm_model_preference = settings.llm_model_arg
        if initial_llm_model_preference and initial_llm_model_preference in settings.available_models:
            settings.current_llm_model = initial_llm_model_preference
        elif settings.available_models and settings.available_models[0] != "fallback/unknown-model":
            if initial_llm_model_preference:
                logger.warning(f"LLM model '{initial_llm_model_preference}' not found. Using first available: {settings.available_models[0]}.")
            settings.current_llm_model = settings.available_models[0]
        else:
            settings.current_llm_model = initial_llm_model_preference or "fallback/unknown-model"
            logger.warning(f"Using specified or fallback LLM model: {settings.current_llm_model}. Availability/cost data might be missing.")
        logger.info(f"Initial LLM model (classic backend) set to: {settings.current_llm_model}")

        if settings.is_openai_tts:
            settings.available_voices_tts = OPENAI_TTS_VOICES
        else:
            settings.available_voices_tts = get_voices(settings.tts_base_url, settings.tts_api_key)
            if not settings.available_voices_tts:
                logger.warning(f"Could not retrieve voices from custom TTS server at {settings.tts_base_url}.")
        logger.info(f"Available TTS voices (classic backend): {settings.available_voices_tts}")

        initial_tts_voice_preference = settings.tts_voice_arg  # Classic TTS voice
        if initial_tts_voice_preference and initial_tts_voice_preference in settings.available_voices_tts:
            settings.current_tts_voice = initial_tts_voice_preference
        elif settings.available_voices_tts:
            if initial_tts_voice_preference:
                 logger.warning(f"Classic TTS voice '{initial_tts_voice_preference}' not found. Using first available: {settings.available_voices_tts[0]}.")
            settings.current_tts_voice = settings.available_voices_tts[0]
        else:
            settings.current_tts_voice = initial_tts_voice_preference
            logger.error(f"No classic TTS voices available or specified voice '{settings.current_tts_voice}' is invalid. TTS may fail.")
        logger.info(f"Initial classic TTS voice set to: {settings.current_tts_voice}")

    # --- Common Post-Backend-Specific Setup ---
    if settings.system_message:
        logger.info(f"Loaded SYSTEM_MESSAGE: '{settings.system_message[:50]}...'")
    else:
        logger.info("No SYSTEM_MESSAGE defined.")

    try:
        app_name = "SimpleVoiceChat"
        app_author = "Attila"
        user_data_dir = Path(platformdirs.user_data_dir(app_name, app_author))
        settings.chat_log_dir = user_data_dir / "chats"
        settings.chat_log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Chat log directory set to: {settings.chat_log_dir}")
    except Exception as e:
        logger.error(f"Failed to create chat log directory: {e}. Chat logging disabled.")
        settings.chat_log_dir = None

    # TTS audio directory setup (primarily for classic backend)
    if settings.backend == "classic":
        try:
            app_name = "SimpleVoiceChat"
            app_author = "Attila"
            try:
                cache_base_dir = Path(platformdirs.user_cache_dir(app_name, app_author))
            except Exception:
                logger.warning("Could not find user cache directory, falling back to user data directory for TTS audio.")
                cache_base_dir = Path(platformdirs.user_data_dir(app_name, app_author))

            settings.tts_base_dir = cache_base_dir / "tts_audio"
            settings.tts_base_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Base TTS audio directory: {settings.tts_base_dir}")

            settings.tts_audio_dir = settings.tts_base_dir / settings.startup_timestamp_str
            settings.tts_audio_dir.mkdir(exist_ok=True)
            logger.info(f"This run's TTS audio directory (classic backend): {settings.tts_audio_dir}")
        except Exception as e:
            logger.error(f"Failed to create temporary TTS audio directory for classic backend: {e}. TTS audio saving might fail.")
    elif settings.backend in {"openai", "gemini"}:  # Add Gemini
        logger.info(f"TTS audio file saving to disk is not applicable for '{settings.backend}' backend.")
        settings.tts_audio_dir = None

    logger.info(f"Application server host: {settings.host}")
    logger.info(
        f"Application server preferred port: {settings.preferred_port}",
    )

    # --- Stream Handler Setup ---
    stream_handler: Any
    if settings.backend == "openai":
        logger.info("Initializing Stream with OpenAIRealtimeHandler.")
        stream_handler = OpenAIRealtimeHandler(app_settings=settings)
    elif settings.backend == "gemini":  # Add Gemini case
        logger.info("Initializing Stream with GeminiRealtimeHandler.")
        stream_handler = GeminiRealtimeHandler(app_settings=settings)
    else:
        logger.info("Initializing Stream with ReplyOnPause handler for classic backend.")
        stream_handler = ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                audio_chunk_duration=3.0,
                started_talking_threshold=0.2,
                speech_threshold=0.2,
            ),
            model_options=SileroVadOptions(
                threshold=0.6,
                min_speech_duration_ms=800,
                min_silence_duration_ms=3500,
            ),
            can_interrupt=True,
        )

    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=stream_handler,
        track_constraints={
            "echoCancellation": True,
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            # Ideal sample rate should match what the handler expects as input.
            # OpenAI expects 24kHz. Gemini expects 16kHz. Classic is flexible.
            "sampleRate": {"ideal": GEMINI_REALTIME_INPUT_SAMPLE_RATE if settings.backend == "gemini" else (OPENAI_REALTIME_SAMPLE_RATE if settings.backend == "openai" else 16000)},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        },
        rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
        concurrency_limit=5 if get_space() else None,
        time_limit=180 if get_space() else None,
    )

    app = FastAPI()
    stream.mount(app)
    register_endpoints(app, stream)

    current_host = settings.host
    preferred_port_val = settings.preferred_port
    actual_port = preferred_port_val
    max_retries = 10

    if is_port_in_use(actual_port, current_host):
        logger.warning(
            f"Preferred port {actual_port} on host {current_host} is in use. Searching for an available port...",
        )
        found_port = False
        for attempt in range(max_retries):
            new_port = random.randint(1024, 65535)
            logger.debug(f"Attempt {attempt + 1}: Checking port {new_port} on {current_host}...")
            if not is_port_in_use(new_port, current_host):
                actual_port = new_port
                found_port = True
                logger.info(f"Found available port: {actual_port} on host {current_host}")
                break
        if not found_port:
            logger.error(
                f"Could not find an available port on host {current_host} after {max_retries} attempts. Exiting.",
            )
            return 1
    else:
        logger.info(f"Using preferred port {actual_port} on host {current_host}")

    settings.port = actual_port
    url = f"http://{current_host}:{actual_port}"

    def run_server() -> None:
        global uvicorn_server
        try:
            config = uvicorn.Config(
                app,
                host=current_host,
                port=actual_port,
                log_config=None,
            )
            uvicorn_server = uvicorn.Server(config)
            logger.info(f"Starting Uvicorn server on {current_host}:{actual_port}...")
            uvicorn_server.run()
            logger.info("Uvicorn server has stopped.")
        except Exception as e:
            logger.critical(f"Uvicorn server encountered an error: {e}")
        finally:
            uvicorn_server = None

    monitor_thread = threading.Thread(target=monitor_heartbeat_thread, daemon=True)
    monitor_thread.start()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logger.debug("Waiting for Uvicorn server to initialize...")
    time.sleep(3.0)

    if not server_thread.is_alive() or uvicorn_server is None:
        logger.critical(
            "Uvicorn server thread failed to start or initialize correctly. Exiting.",
        )
        return 1
    logger.debug("Server thread appears to be running.")

    exit_code = 0
    try:
        if settings.browser:
            logger.info(f"Opening application in default web browser at: {url}")
            webbrowser.open(url, new=1)
            logger.info(
                "Application opened in browser. Server is running in the background.",
            )
            logger.info("Press Ctrl+C to stop the server.")
            try:
                server_thread.join()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down.")
            finally:
                logger.info("Signaling heartbeat monitor thread to stop...")
                shutdown_event.set()
                if (
                    uvicorn_server
                    and server_thread.is_alive()
                    and not uvicorn_server.should_exit
                ):
                    logger.info("Signaling Uvicorn server to shut down...")
                    uvicorn_server.should_exit = True
                elif uvicorn_server and uvicorn_server.should_exit:
                    logger.info("Uvicorn server already signaled to shut down.")
                elif not server_thread.is_alive():
                    logger.info("Server thread already stopped.")
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown.",
                    )
                if server_thread.is_alive():
                    logger.info("Waiting for Uvicorn server thread to join...")
                    server_thread.join(timeout=5.0)
                    if server_thread.is_alive():
                        logger.warning(
                            "Uvicorn server thread did not exit gracefully after 5 seconds.",
                        )
                    else:
                        logger.info("Uvicorn server thread joined successfully.")
                logger.info("Waiting for heartbeat monitor thread to join...")
                monitor_thread.join(timeout=2.0)
                if monitor_thread.is_alive():
                    logger.warning(
                        "Heartbeat monitor thread did not exit gracefully after 2 seconds.",
                    )
                else:
                    logger.info("Heartbeat monitor thread joined successfully.")
        else:
            logger.info(f"Creating pywebview window for URL: {url}")
            api = Api(None)
            webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
            logger.info("pywebview setting OPEN_DEVTOOLS_IN_DEBUG set to False.")

            pywebview_window = webview.create_window(
                f"Simple Voice Chat v{APP_VERSION}",
                url,
                width=800,
                height=800,
                js_api=api,
            )
            api._window = pywebview_window

            logger.info("Starting pywebview...")
            try:
                webview.start(debug=True, gui="qt")
            except Exception as e:
                logger.critical(f"Pywebview encountered an error: {e}")
                exit_code = 1
            finally:
                logger.info("Pywebview window closed or heartbeat timed out.")
                logger.info("Signaling heartbeat monitor thread to stop...")
                shutdown_event.set()
                if uvicorn_server and not uvicorn_server.should_exit:
                    logger.info("Signaling Uvicorn server to shut down...")
                    uvicorn_server.should_exit = True
                elif uvicorn_server and uvicorn_server.should_exit:
                    logger.info("Uvicorn server already signaled to shut down.")
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown.",
                    )
                logger.info("Waiting for Uvicorn server thread to join...")
                server_thread.join(timeout=5.0)
                if server_thread.is_alive():
                    logger.warning(
                        "Uvicorn server thread did not exit gracefully after 5 seconds.",
                    )
                else:
                    logger.info("Uvicorn server thread joined successfully.")
                logger.info("Waiting for heartbeat monitor thread to join...")
                monitor_thread.join(timeout=2.0)
                if monitor_thread.is_alive():
                    logger.warning(
                        "Heartbeat monitor thread did not exit gracefully after 2 seconds.",
                    )
                else:
                    logger.info("Heartbeat monitor thread joined successfully.")
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred in the main execution block: {e}",
            exc_info=True,
        )
        exit_code = 1
        shutdown_event.set()
        if uvicorn_server and not uvicorn_server.should_exit:
            uvicorn_server.should_exit = True

    logger.info(f"Main function returning exit code: {exit_code}")
    return exit_code

