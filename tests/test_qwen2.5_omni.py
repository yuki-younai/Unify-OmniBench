import os
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/apdcephfs_hldy/share_304318596/weiyangguo/models/Qwen2.5-Omni-3B"

# Local test data paths (relative to this script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(SCRIPT_DIR, "..", "example")
VIDEO_PATH = os.path.join(EXAMPLE_DIR, "draw.mp4")
AUDIO_PATH = os.path.join(EXAMPLE_DIR, "cough.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

USE_AUDIO_IN_VIDEO = True

# ============================================================
# Load Model & Processor
# ============================================================
print("=" * 60)
print("Loading model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("Model loaded successfully.\n")


# ============================================================
# Helper: run single-turn inference
# ============================================================
def run_inference(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=True, speaker=None):
    """
    Generic single-turn inference.
    Returns (text_output, audio_tensor_or_None).
    """
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    gen_kwargs = {**inputs, "use_audio_in_video": use_audio_in_video, "return_audio": return_audio}
    if speaker is not None:
        gen_kwargs["speaker"] = speaker

    if return_audio:
        text_ids, audio = model.generate(**gen_kwargs)
    else:
        text_ids = model.generate(**gen_kwargs)
        audio = None

    text_output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text_output, audio


# ============================================================
# Test 1: Video input (with audio) → Text + Speech output
# ============================================================
print("=" * 60)
print("Test 1: Video input → Text + Speech")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "video", "video": VIDEO_PATH}]},
]
text_out, audio_out = run_inference(conversation, use_audio_in_video=True)
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test1_video_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio saved to test1_video_output.wav\n")


# ============================================================
# Test 2: Audio-only input → Text + Speech
# ============================================================
print("=" * 60)
print("Test 2: Audio-only input → Text + Speech")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "audio", "audio": AUDIO_PATH}, {"type": "text", "text": "What sound is this?"}]},
]
text_out, audio_out = run_inference(conversation)
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test2_audio_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio saved to test2_audio_output.wav\n")


# ============================================================
# Test 3: Text-only input → Text + Speech
# ============================================================
print("=" * 60)
print("Test 3: Text-only input → Text + Speech")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": "who are you?"},
]
text_out, audio_out = run_inference(conversation)
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test3_text_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio saved to test3_text_output.wav\n")


# ============================================================
# Test 4: Text-only → return_audio=False (faster text-only response)
# ============================================================
print("=" * 60)
print("Test 4: Text-only → return_audio=False")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": "What is 2 + 3?"},
]
text_out, _ = run_inference(conversation, return_audio=False)
print(f"  Text: {text_out}\n")


# ============================================================
# Test 5: Change voice type (Chelsie / Ethan)
# ============================================================
print("=" * 60)
print("Test 5: Change voice type to Ethan")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": "Say hello in a cheerful way."},
]
text_out, audio_out = run_inference(conversation, speaker="Ethan")
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test5_ethan_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio (Ethan voice) saved to test5_ethan_output.wav\n")


# ============================================================
# Test 6: Mixed media (video + audio + text)
# ============================================================
print("=" * 60)
print("Test 6: Mixed media input → Text + Speech")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "audio", "audio": AUDIO_PATH},
        {"type": "text", "text": "Describe what you see in the video and what you hear in the audio."},
    ]},
]
text_out, audio_out = run_inference(conversation, use_audio_in_video=True)
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test6_mixed_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio saved to test6_mixed_output.wav\n")


# ============================================================
# Test 7: Video without audio → Text + Speech
# ============================================================
print("=" * 60)
print("Test 7: Video (ignore audio track) → Text + Speech")
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "video", "video": VIDEO_PATH}, {"type": "text", "text": "What is happening in this video?"}]},
]
text_out, audio_out = run_inference(conversation, use_audio_in_video=False)
print(f"  Text: {text_out}")
sf.write(os.path.join(OUTPUT_DIR, "test7_video_noaudio_output.wav"), audio_out.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print("  Audio saved to test7_video_noaudio_output.wav\n")


# ============================================================
# Test 8: Batch inference (multiple conversations, return_audio=False)
# ============================================================
print("=" * 60)
print("Test 8: Batch inference (return_audio=False)")

conversation1 = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": "who are you?"},
]
conversation2 = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "video", "video": VIDEO_PATH}, {"type": "text", "text": "Describe this video briefly."}]},
]
conversation3 = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "audio", "audio": AUDIO_PATH}, {"type": "text", "text": "What sound do you hear?"}]},
]
conversations = [conversation1, conversation2, conversation3]

text_batch = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
inputs = processor(
    text=text_batch, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=True,
)
inputs = inputs.to(model.device).to(model.dtype)

text_ids = model.generate(**inputs, use_audio_in_video=True, return_audio=False)
text_outputs = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for i, t in enumerate(text_outputs):
    print(f"  Batch[{i}]: {t}")
print()


# ============================================================
# Test 9: disable_talker() → save memory, text-only output
# ============================================================
print("=" * 60)
print("Test 9: disable_talker() → text-only")
model.disable_talker()
conversation = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": "What is the capital of France?"},
]
text_out, _ = run_inference(conversation, return_audio=False)
print(f"  Text: {text_out}\n")


print("=" * 60)
print("All tests completed!")





