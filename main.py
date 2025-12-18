import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?"

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="/home/hsiangnianian/gitproject/HsiangNianian/agentio/assets/audio/c337ef95-0587-4ebd-8205-87ce09f4dbb6.wav")

ta.save("test-turbo.wav", wav, model.sr)
