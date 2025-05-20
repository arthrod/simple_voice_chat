import os

from src.simple_voice_chat import main

if __name__ == "__main__":
    # Example arguments:
    # Replace these with your desired configuration
    args = [
        "--backend", "gemini",
        "--llm-model", "gemini-2.0-flash-live-001",
        "--tts-voice", "Aoede",
        "--stt-language", "pt",
        "--browser",  # Launch in browser instead of pywebview GUI
        # Add other arguments as needed, like:
        # "--openai-api-key", "YOUR_OPENAI_KEY_HERE", # If using OpenAI backend
        "--llm-api-key", os.environ["GOOGLE_API_KEY"],    # If classic backend needs a key for LLM
        # "--stt-api-key", "YOUR_STT_KEY_HERE",    # If classic backend STT needs a key
        # "--tts-api-key", "YOUR_TTS_KEY_HERE",    # If classic backend TTS needs a key
    ]

    # The main function expects a list of strings, similar to sys.argv
    # It's decorated with @click.command(), so we call it with .main(args)
    # or by directly invoking it if click handles parsing internally when called this way.
    # For programmatic invocation with click, it's often easier to let click parse:
    import os
    # To ensure LiteLLM runs in production mode if not already set by the main script early enough
    os.environ["LITELLM_MODE"] = "PRODUCTION"

    # Call the click command directly
    # Note: click commands usually expect to be called as if from the command line.
    # To pass arguments programmatically to a click command, you typically invoke `main.main(args=args_list, standalone_mode=False)`.
    # However, since `main` is already a click command, we can try to directly invoke it.
    # If `main()` is defined as `def main(): @click.pass_context def cli(ctx, ...)` then `main(args)` works.
    # If `def main(...)` is the click command itself, it consumes args from sys.argv by default.
    # The `main` function in src.py is a click command itself: `@click.command(...) def main(...)`
    # So, to run it programmatically as if from CLI:
    try:
        # sys.argv needs to be manipulated if click is to parse it automatically,
        # or use the programmatic API if available.
        # The simplest way with click
        main.main(args=args, standalone_mode=False)
    except SystemExit:
        # Click commands often call sys.exit(). We can catch this if running in a script.
        pass
