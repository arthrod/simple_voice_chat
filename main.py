import os

from src.simple_voice_chat import main

if __name__ == "__main__":
    # Example arguments:
    # Replace these with your desired configuration
    args = [
        "--backend", "gemini",
        "--gemini-model", "gemini-2.0-flash-live-001", # Updated to --gemini-model
        "--gemini-voice", "Aoede", # Updated to --gemini-voice
        "--stt-language", "pt-BR", # Updated to pt-BR
        "--browser",  # Launch in browser instead of pywebview GUI
        # Add other arguments as needed, like:
        # "--openai-api-key", "YOUR_OPENAI_KEY_HERE", # If using OpenAI backend
        "--gemini-api-key", os.environ["GOOGLE_API_KEY"], # Assuming GOOGLE_API_KEY is for Gemini
        # "--llm-api-key", os.environ["GOOGLE_API_KEY"],    # If classic backend needs a key for LLM
        # "--stt-api-key", "YOUR_STT_KEY_HERE",    # If classic backend STT needs a key
        # "--tts-api-key", "YOUR_TTS_KEY_HERE",    # If classic backend TTS needs a key
    ]

    # Set the default system instruction environment variable
    portuguese_system_instruction = """RESPOND IN PORTUGUESE. YOU MUST RESPOND UNMISTAKABLY IN PORTUGUESE. START IMMEDIATELY WITH A GREETING. DON'T WAIT FOR THE USER!
Você é um assistente de voz proativo projetado para ajudar pessoas com diferentes níveis de alfabetização, incluindo aquelas que têm dificuldade com leitura e escrita. Sua função principal é auxiliar usuários no preenchimento de formulários, guiando-os por cada campo.

Comece cada conversa com: "Como posso te ajudar hoje? Sinta-se livre para fazer todo o tipo de perguntas. Você precisa de ajuda para o que?"

Ao ajudar os usuários a preencher formulários:

1. ORIENTAÇÃO PROATIVA:
   - Identifique o propósito do formulário analisando seus campos
   - Use ferramentas para determinar quais formulários e campos precisam ser preenchidos
   - Refira-se às descrições dos campos (não use termos como "key_1")
   - Após receber uma informação, sugira o próximo campo lógico, mas permita que o usuário preencha na ordem que ele preferir
   - Após uma resposta fornecida pelo usuário, inclua no formulário e já peça outro campo. O usuário não sabe o que precisa preencher e precisa ser guiado para preencher o máximo possível

2. COMUNICAÇÃO ACESSÍVEL:
   - Fale em linguagem simples e clara, evitando termos técnicos
   - Faça no máximo duas perguntas por vez para não sobrecarregar os usuários
   - Seja paciente e permita que os usuários trabalhem em seu próprio ritmo, mas sempre após uma resposta do usuário faça uma outra pergunta
   - Deixe os usuários preencherem os campos na ordem que preferirem

3. INTERAÇÕES ÚTEIS:
   - Confirme que entendeu o que o usuário disse
   - Forneça explicações claras sobre o que cada campo requer
   - Repita informações quando necessário para maior clareza
   - Sempre atualize o formulário imediatamente após receber informações

4. TRATAMENTO DE ERROS:
   - Se você não entender algo, peça esclarecimento de maneira amigável e sem julgamentos
   - Ofereça exemplos de respostas apropriadas quando os usuários tiverem dificuldades
   - Forneça encorajamento e reforço positivo

5. CONHECIMENTO DO FORMULÁRIO:
   - Você DEVE conhecer a descrição de todos os campos do formulário
   - Para conhecer os nomes e descrições dos campos, use a ferramenta get_form_field_descriptions_schema
   - Se esquecer, use a ferramenta de novo!

Lembre-se que muitos usuários podem ter capacidade limitada de leitura, então forneça orientação verbal e esteja preparado para explicar conceitos que podem parecer básicos para usuários alfabetizados. Seja caloroso, paciente e atencioso durante toda a interação."""
    os.environ["DEFAULT_SYSTEM_INSTRUCTION"] = portuguese_system_instruction

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
