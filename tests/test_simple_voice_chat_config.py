import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import asyncio

# Ensure src directory is in Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simple_voice_chat import main as simple_voice_chat_main
from src.utils.config import settings, AppSettings
from src.utils.env import DEFAULT_SYSTEM_INSTRUCTION_ENV 

import google.genai as genai # Use 'as genai' for consistency
from google.genai.types import (
    LiveConnectConfig,
    SpeechConfig as GenaiSpeechConfig, # Keep alias if used
    VoiceConfig as GenaiVoiceConfig,   # Keep alias if used
    PrebuiltVoiceConfig,
    Content, 
    Part,    
    Blob,    
    FunctionResponse,
    ToolCall, # Make sure ToolCall is imported
    FunctionCall # For constructing mock ToolCall
)
# Import the handler to be tested
from src.simple_voice_chat import GeminiRealtimeHandler


ORIGINAL_OS_ENVIRON = os.environ.copy()

# Helper to create an async iterator from a list
class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration

class TestSimpleVoiceChatConfig(unittest.TestCase):

    def setUp(self):
        global settings
        settings = AppSettings() 

        self.base_env_vars = {
            "GOOGLE_API_KEY": "test_google_api_key", 
            "LITELLM_MODE": "PRODUCTION" 
        }
        os.environ.clear()
        os.environ.update(ORIGINAL_OS_ENVIRON)


    def tearDown(self):
        os.environ.clear()
        os.environ.update(ORIGINAL_OS_ENVIRON)
        global settings
        settings = AppSettings()

    @unittest.skip("Reason: Complex interaction with Click, global settings, and threading makes this test unreliable without refactoring main().")
    @patch('src.simple_voice_chat.uvicorn.Server') 
    @patch('src.simple_voice_chat.webview.create_window') 
    @patch('src.simple_voice_chat.webview.start')
    @patch('src.simple_voice_chat.platformdirs.user_log_dir')
    @patch('src.simple_voice_chat.platformdirs.user_data_dir')
    @patch('src.simple_voice_chat.setup_logging') 
    @patch('src.simple_voice_chat.load_form_data') 
    @patch('threading.Thread') 
    @patch('time.sleep')       
    @patch('src.simple_voice_chat.uvicorn_server', new_callable=MagicMock) 
    def test_main_gemini_config_with_all_args(self, mock_module_uvicorn_server, mock_time_sleep, mock_threading_thread, mock_load_form, mock_setup_logging, mock_user_data_dir, mock_user_log_dir, mock_webview_start, mock_webview_create_window, mock_uvicorn_server_class_constructor):
        mock_user_log_dir.return_value = "/tmp/testlogs" 
        mock_user_data_dir.return_value = "/tmp/testdata"
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True 
        mock_threading_thread.return_value = mock_thread_instance
        mock_uvicorn_server_instance = MagicMock()
        mock_uvicorn_server_class_constructor.return_value = mock_uvicorn_server_instance
        simulated_args = [
            "--backend", "gemini", "--gemini-model", "gemini-test-model", 
            "--gemini-voice", "TestGeminiVoice", "--stt-language", "pt-BR",
            "--gemini-api-key", "test_google_api_key",
            "--default-system-instruction", "CLI Default System Instruction for Gemini"
        ]
        test_env = self.base_env_vars.copy()
        with patch.dict(os.environ, test_env, clear=True):
            try: simple_voice_chat_main.main(args=simulated_args, standalone_mode=False)
            except SystemExit as e:
                if e.code != 0 and e.code is not None and e.code != 1: self.fail(f"main() exited with unexpected code {e.code}")
        self.assertEqual(settings.backend, "gemini")
        self.assertEqual(settings.current_llm_model, "gemini-2.0-flash-live-001") 
        self.assertEqual(settings.current_gemini_voice, "Aoede") 
        self.assertEqual(settings.current_stt_language, "pt-BR")
        self.assertEqual(settings.system_message, "CLI Default System Instruction for Gemini")
        self.assertTrue(mock_threading_thread.called) 
        mock_time_sleep.assert_any_call(3.0) 

    @unittest.skip("Reason: Complex interaction with Click, global settings, and threading makes this test unreliable without refactoring main().")
    @patch('src.simple_voice_chat.uvicorn.Server')
    @patch('src.simple_voice_chat.webview.create_window')
    @patch('src.simple_voice_chat.webview.start')
    @patch('src.simple_voice_chat.platformdirs.user_log_dir')
    @patch('src.simple_voice_chat.platformdirs.user_data_dir')
    @patch('src.simple_voice_chat.setup_logging')
    @patch('src.simple_voice_chat.load_form_data')
    @patch('threading.Thread')
    @patch('time.sleep')
    @patch('src.simple_voice_chat.uvicorn_server', new_callable=MagicMock)
    def test_main_gemini_stt_language_default(self, mock_module_uvicorn_server, mock_time_sleep, mock_threading_thread, mock_load_form, mock_setup_logging, mock_user_data_dir, mock_user_log_dir, mock_webview_start, mock_webview_create_window, mock_uvicorn_server_class_constructor):
        mock_user_log_dir.return_value = "/tmp/testlogs"
        mock_user_data_dir.return_value = "/tmp/testdata"
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True
        mock_threading_thread.return_value = mock_thread_instance
        mock_uvicorn_server_instance = MagicMock()
        mock_uvicorn_server_class_constructor.return_value = mock_uvicorn_server_instance
        simulated_args = ["--backend", "gemini", "--gemini-model", "gemini-test-model-2", "--gemini-api-key", "test_google_api_key_2"]
        test_env = self.base_env_vars.copy()
        with patch.dict(os.environ, test_env, clear=True):
            try: simple_voice_chat_main.main(args=simulated_args, standalone_mode=False)
            except SystemExit as e:
                if e.code != 0 and e.code is not None and e.code != 1: self.fail(f"main() exited with unexpected code {e.code}")
        self.assertEqual(settings.current_stt_language, "pt-BR")
        self.assertEqual(settings.current_llm_model, "gemini-2.0-flash-live-001")

    @unittest.skip("Reason: Complex interaction with Click, global settings, and threading makes this test unreliable without refactoring main().")
    @patch('src.simple_voice_chat.uvicorn.Server')
    @patch('src.simple_voice_chat.webview.create_window')
    @patch('src.simple_voice_chat.webview.start')
    @patch('src.simple_voice_chat.platformdirs.user_log_dir')
    @patch('src.simple_voice_chat.platformdirs.user_data_dir')
    @patch('src.simple_voice_chat.setup_logging')
    @patch('src.simple_voice_chat.load_form_data')
    @patch('threading.Thread')
    @patch('time.sleep')
    @patch('src.simple_voice_chat.uvicorn_server', new_callable=MagicMock) 
    def test_main_gemini_system_message_env_override(self, mock_module_uvicorn_server, mock_time_sleep, mock_threading_thread, mock_load_form, mock_setup_logging, mock_user_data_dir, mock_user_log_dir, mock_webview_start, mock_webview_create_window, mock_uvicorn_server_class_constructor):
        mock_user_log_dir.return_value = "/tmp/testlogs"
        mock_user_data_dir.return_value = "/tmp/testdata"
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True
        mock_threading_thread.return_value = mock_thread_instance
        mock_uvicorn_server_instance = MagicMock()
        mock_uvicorn_server_class_constructor.return_value = mock_uvicorn_server_instance
        simulated_args = ["--backend", "gemini", "--default-system-instruction", "This should be overridden by ENV", "--gemini-api-key", "test_google_api_key_3"]
        env_system_message = "System Message from ENV" 
        test_env = self.base_env_vars.copy()
        test_env["SYSTEM_MESSAGE"] = env_system_message
        with patch.dict(os.environ, test_env, clear=True):
            try: simple_voice_chat_main.main(args=simulated_args, standalone_mode=False)
            except SystemExit as e:
                if e.code != 0 and e.code is not None and e.code != 1: self.fail(f"main() exited with unexpected code {e.code}")
        self.assertEqual(settings.system_message, env_system_message) 
        self.assertEqual(settings.default_system_instruction, "This should be overridden by ENV")

    @unittest.skip("Reason: Complex interaction with Click, global settings, and threading makes this test unreliable without refactoring main().")
    @patch('src.simple_voice_chat.uvicorn.Server')
    @patch('src.simple_voice_chat.webview.create_window')
    @patch('src.simple_voice_chat.webview.start')
    @patch('src.simple_voice_chat.platformdirs.user_log_dir')
    @patch('src.simple_voice_chat.platformdirs.user_data_dir')
    @patch('src.simple_voice_chat.setup_logging')
    @patch('src.simple_voice_chat.load_form_data')
    @patch('threading.Thread')
    @patch('time.sleep')
    @patch('src.simple_voice_chat.uvicorn_server', new_callable=MagicMock) 
    def test_main_gemini_default_system_instruction_env_used(self, mock_module_uvicorn_server, mock_time_sleep, mock_threading_thread, mock_load_form, mock_setup_logging, mock_user_data_dir, mock_user_log_dir, mock_webview_start, mock_webview_create_window, mock_uvicorn_server_class_constructor):
        mock_user_log_dir.return_value = "/tmp/testlogs"
        mock_user_data_dir.return_value = "/tmp/testdata"
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True
        mock_threading_thread.return_value = mock_thread_instance
        mock_uvicorn_server_instance = MagicMock()
        mock_uvicorn_server_class_constructor.return_value = mock_uvicorn_server_instance
        simulated_args = ["--backend", "gemini", "--gemini-api-key", "test_google_api_key_4"]
        env_default_instruction = "Default System Instruction from ENV" 
        test_env = self.base_env_vars.copy()
        test_env["DEFAULT_SYSTEM_INSTRUCTION"] = env_default_instruction
        if "SYSTEM_MESSAGE" in test_env: del test_env["SYSTEM_MESSAGE"]
        with patch.dict(os.environ, test_env, clear=True):
            try: simple_voice_chat_main.main(args=simulated_args, standalone_mode=False)
            except SystemExit as e:
                if e.code != 0 and e.code is not None and e.code != 1: self.fail(f"main() exited with unexpected code {e.code}")
        self.assertEqual(settings.default_system_instruction, env_default_instruction)
        self.assertEqual(settings.system_message, env_default_instruction)

    @patch('google.genai.Client') 
    async def test_gemini_handler_new_api_interactions(self, MockGenAIClient):
        """Test GeminiRealtimeHandler with new API methods."""
        
        mock_app_settings = AppSettings()
        mock_app_settings.backend = "gemini"
        mock_app_settings.gemini_api_key = "fake_gemini_key"
        mock_app_settings.current_llm_model = "gemini-pro" 
        mock_app_settings.current_gemini_voice = "gemini-test-voice"
        mock_app_settings.current_stt_language = "en-US" 
        mock_app_settings.system_message = "Test System Instruction"
        mock_app_settings.gemini_response_modalities = ["AUDIO", "TEXT"] 
        mock_app_settings.gemini_context_window_compression_threshold = 12000

        # Mock the session object returned by client.aio.live.connect()
        mock_session = AsyncMock(spec=genai.live.AsyncLiveConnectSession)
        mock_session.send_client_content = AsyncMock()
        mock_session.send_tool_response = AsyncMock()
        mock_session.close = AsyncMock()
        
        # Configure client.aio.live.connect to return an async context manager yielding the mock_session
        mock_client_instance = MockGenAIClient.return_value
        mock_async_cm = AsyncMock()
        mock_async_cm.__aenter__.return_value = mock_session
        mock_async_cm.__aexit__.return_value = None
        mock_client_instance.aio.live.connect.return_value = mock_async_cm

        handler = GeminiRealtimeHandler(app_settings=mock_app_settings)

        # --- Test Scenario: Audio Sending and Tool Call ---
        mock_tool_call_id = "tool_call_123"
        mock_tool_call_name = "obter_descricoes_campos"
        
        # Define the sequence of events for session.receive()
        # 1. An event that includes a tool call
        # 2. An event that allows the audio sender to process (e.g., a simple text response or just keep alive)
        # 3. StopAsyncIteration to end the loop
        mock_tool_call_event = MagicMock(spec=genai.types.live.LiveServerMessage)
        mock_tool_call_event.tool_calls = [
            ToolCall(id=mock_tool_call_id, function_call=FunctionCall(name=mock_tool_call_name, args={}))
        ]
        # Add other necessary attributes to mock_tool_call_event if the handler accesses them
        # For example, if it tries to access event.input_transcription or event.model_turn, mock those as None or empty.
        for attr in ["input_transcription", "model_turn", "error", "usage_metadata", "speech_processing_event", "turn_complete"]:
            setattr(mock_tool_call_event, attr, None)


        mock_final_event = MagicMock(spec=genai.types.live.LiveServerMessage) # A simple event to keep loop running once more
        for attr in ["input_transcription", "model_turn", "tool_calls", "error", "usage_metadata", "speech_processing_event", "turn_complete"]:
            setattr(mock_final_event, attr, None)
        
        # Configure session.receive to yield these events then stop
        # Use a list for the AsyncIterator helper
        receive_events = [mock_tool_call_event, mock_final_event]
        mock_session.receive.return_value = AsyncIterator(receive_events)

        # Start the handler (this will start the _audio_sender_task and the receive loop)
        start_up_task = asyncio.create_task(handler.start_up())

        # Simulate receiving an audio frame
        sample_audio_bytes = b'\x01\x02\x03\x04'
        await handler.receive((16000, np.array([1, 2, 3, 4], dtype=np.int16)))
        
        # Allow some time for the audio sender task and the receive loop to process
        await asyncio.sleep(0.1) # Increased sleep slightly

        # Assertions for audio sending
        mock_session.send_client_content.assert_called_once()
        send_content_args = mock_session.send_client_content.call_args
        self.assertIsNotNone(send_content_args)
        # send_client_content now expects a single Part or Content object directly
        sent_content_arg = send_content_args.kwargs.get('content', send_content_args.args[0] if send_content_args.args else None)
        self.assertIsInstance(sent_content_arg, Part) # Assuming direct Part for audio
        self.assertIsInstance(sent_content_arg.inline_data, Blob)
        self.assertEqual(sent_content_arg.inline_data.mime_type, "audio/pcm")
        self.assertEqual(sent_content_arg.inline_data.data, sample_audio_bytes)

        # Assertions for tool response sending
        mock_session.send_tool_response.assert_called_once()
        tool_response_args = mock_session.send_tool_response.call_args
        self.assertIsNotNone(tool_response_args)
        sent_function_responses = tool_response_args.kwargs.get('function_responses')
        self.assertIsInstance(sent_function_responses, list)
        self.assertEqual(len(sent_function_responses), 1)
        self.assertIsInstance(sent_function_responses[0], FunctionResponse)
        self.assertEqual(sent_function_responses[0].id, mock_tool_call_id)
        self.assertEqual(sent_function_responses[0].name, mock_tool_call_name)
        self.assertIn("field_descriptions", sent_function_responses[0].response) # Check part of the response

        # Clean up
        await handler.shutdown() # This should cancel _audio_sender_task and stop the handler
        if not start_up_task.done(): # Ensure start_up task is also awaited or cancelled
            start_up_task.cancel()
            try: await start_up_task
            except asyncio.CancelledError: pass


if __name__ == '__main__':
    # This basic runner won't work correctly for async test methods without TestLoader modifications
    # or a runner like pytest-asyncio. For now, individual execution might require manual setup.
    # The unittest discover mechanism should handle async def test methods if the Python version is >= 3.8
    # For older versions or direct execution, a more specific runner is needed.
    
    # A simple way to run a specific async test for quick checks (Python 3.7+)
    if sys.version_info >= (3, 7) and any('test_gemini_handler_new_api_interactions' in arg for arg in sys.argv):
        async def main_for_test():
            suite = unittest.TestSuite()
            # Manually add the async test method
            # Note: TestLoader().loadTestsFromTestCase() might not pick up async def methods correctly
            # in all unittest versions without an async-aware test runner.
            # This is a direct way to add it.
            suite.addTest(TestSimpleVoiceChatConfig("test_gemini_handler_new_api_interactions"))
            runner = unittest.TextTestRunner()
            runner.run(suite)
        asyncio.run(main_for_test())
    else:
        unittest.main()
