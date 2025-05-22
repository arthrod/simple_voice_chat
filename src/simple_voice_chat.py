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
# ... (monitor_heartbeat_thread function - unchanged) ...
# --- Main Click Command ---
# ... (main function - unchanged) ...
