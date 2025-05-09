# Simple Voice Chat

This project provides a flexible voice chat interface that connects to various Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) services.

![Screenshot](screenshot.png)

**Acknowledgement:** This project heavily relies on the fantastic [fastrtc](https://github.com/gradio-app/fastrtc) library, which simplifies real-time audio streaming over WebRTC, making this application possible.

## Motivation

This project aims to provide a versatile and cost-effective voice chat interface. While initially driven by the desire for alternatives to OpenAI's real-time voice API, it has evolved to offer multiple backend options, including direct integration with OpenAI's real-time services. This allows users to choose the best STT, LLM, and TTS combination for their needs, whether prioritizing cost, performance, self-hosting, or specific provider features.

## Features

*   🚀 **Multiple Backends:**
    *   **Classic Backend:** Offers a modular approach, allowing you to connect separate services for:
        *   🗣️ **STT:** Supports OpenAI Whisper API or self-hosted engines like [Speaches](https://github.com/speaches-ai/speaches) (Faster Whisper).
        *   🧠 **LLM:** Integrates with [LiteLLM](https://github.com/BerriAI/litellm), enabling connections to OpenAI, Anthropic, Google, Mistral, Cohere, Azure, local models (via LiteLLM proxy, vLLM, Ollama), and more.
        *   🔊 **TTS:** Supports OpenAI TTS API or alternatives like [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI).
    *   **OpenAI Backend:** Utilizes OpenAI's real-time voice API for a streamlined, all-in-one voice interaction experience.
    *   🔜 **Gemini Live API (Planned):** Future support for Google's Gemini Live API is planned, offering another advanced real-time voice option.
*   ⚙️ **Highly Configurable:** Adjust backend type, STT/LLM/TTS hosts, ports, models, API keys, STT confidence thresholds (classic backend), TTS voice/speed (classic backend), system messages, and more via CLI arguments or `.env` file.
*   🌐 **Web Interface:** Simple and responsive UI built with HTML, CSS, and JavaScript.
*   📊 **Cost Tracking:**
    *   **Classic Backend:** Real-time cost estimation for OpenAI LLM and TTS usage.
    *   **OpenAI Backend:** Real-time cost estimation based on token usage for the selected OpenAI real-time model.
*   ⚡ **Real-time Interaction:** Low-latency voice communication powered by [fastrtc](https://github.com/gradio-app/fastrtc) (WebRTC).
*   👂 **STT Confidence Filtering (Classic Backend):** Automatically reject low-confidence transcriptions based on configurable thresholds (no speech probability, average log probability, minimum word count).
*   🎤 **Dynamic Settings Adjustment:**
    *   **Classic Backend:** Change LLM model, TTS voice, TTS speed, and STT language on-the-fly.
    *   **OpenAI Backend:** Change STT language and output voice (if supported by the model/API) on-the-fly.
*   🔍 **Fuzzy Search:** Quickly find models and voices using fuzzy search in the UI dropdowns.
*   💬 **System Message Support:** Define a custom system message to guide the LLM's behavior.
*   📝 **Chat History Logging:** Automatically saves conversation history to timestamped JSON files.
*   🔄 **TTS Audio Replay (Classic Backend):** Replay the audio for any assistant message directly from the chat interface.
*   ⌨️ **Keyboard Shortcuts:** Control mute (M), clear chat (Ctrl+R), and toggle options (Shift+S) using keyboard shortcuts.
*   💓 **Connection Monitoring:** Uses a heartbeat mechanism to detect disconnected clients and potentially shut down the server.
*   🖥️ **Cross-Platform GUI:** Runs as a standalone desktop application using `pywebview` (default) or in a standard web browser (`--browser` flag). The application explicitly uses the QT backend for `pywebview` as the GTK backend lacks necessary WebRTC support.

## Installation


1.  Clone the repository:

    ```bash

    git clone https://github.com/thiswillbeyourgithub/simple_voice_chat

    cd simple_voice_chat

    ```

2.  Install the Python packages:

    ```bash

    uv pip install -e .

    ```

3.  (Optional) Configure services using environment variables. You can create a `.env` file based on the available options (see `--help` or `utils/env.py`).



## Usage



Run the main script using Python:


```bash
simple-voice-chat --help
```

The application will start a web server and attempt to open the interface in a dedicated window (or browser tab if `--browser` is specified).

You can choose the backend using the `--backend` option:
*   `--backend classic` (default): Uses separate STT, LLM, and TTS services.
*   `--backend openai`: Uses OpenAI's real-time voice API. Requires `--openai-api-key`.

**For a detailed list of all configuration options, please use the `--help` flag:**

```bash
simple-voice-chat --help
```

This will provide the most up-to-date information on available arguments and their corresponding environment variables, including options specific to each backend.

---



*This README was generated with assistance from [aider.chat](https://aider.chat).*
