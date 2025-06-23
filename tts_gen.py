import time
from pathlib import Path
from openai import OpenAI
import re
import os
import hashlib

# ====================
# ГИПЕРПАРАМЕТРЫ
# ====================
MODEL_NAME = "gpt-4o-mini-tts"  # Модель TTS
WAKE_WORDS = ["Вася", "Вася!", "Вася?", "Вася...", "Эй, Вася"]  # Слова активации
VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]  # Голоса
INSTRUCTIONS = [
    "Voice: Neutral, articulate, focused on clarity above all. Tone: Direct and instructional. Pacing: Deliberate, slightly slower for precision. Pronunciation: Exaggerated consonants, crystal-clear vowels, no vocal fry. Key: Maximum intelligibility for machine listening.",
    "Voice: Gentle, approachable, slightly higher pitch. Tone: Friendly and helpful, like a patient guide. Pacing: Natural, unhurried flow. Pronunciation: Soft consonants, smooth transitions, warm resonance. Key: Comforting clarity without sharpness.",
    "Voice: Bright, upbeat, slightly faster tempo. Tone: Positive and encouraging. Pacing: Lively but controlled. Pronunciation: Crisp plosives (p, t, k), clear vowel articulation, light pitch variation. Key: Clarity with infectious energy.",
    "Voice: Deep, resonant, steady. Tone: Confident, reassuring, slightly formal. Pacing: Measured and deliberate. Pronunciation: Full vowels, smooth transitions, controlled sibilants (s, sh). Key: Unwavering clarity with gravitas.",
    "Voice: Completely flat affect, monotone. Tone: Objective, factual, devoid of emotion. Pacing: Consistent, robotic rhythm. Pronunciation: Machine-like precision, equal stress on syllables. Key: Baseline clarity for algorithmic comparison.",
    "Voice: Casual, conversational, natural rasp. Tone: Relatable and informal. Pacing: Relaxed, slight variations in speed. Pronunciation: Slightly softened consonants, natural vowel reductions, authentic flow. Key: Clear but natural, everyday speech.",
    "Voice: Soft, soothing, lower volume. Tone: Kind, understanding, reassuring. Pacing: Slow, deliberate, with slight pauses. Pronunciation: Gentle articulation, careful enunciation, focus on word shape. Key: Clarity delivered with calmness.",
    "Voice: Polished, articulate, moderate pitch. Tone: Efficient, competent, courteous. Pacing: Steady, purposeful. Pronunciation: Precise consonants, neutral vowels, clean word boundaries. Key: Professional clarity without warmth/coldness.",
    "Voice: Clean, efficient, slightly clipped. Tone: Focused, no-nonsense. Pacing: Brisk but not rushed. Pronunciation: Short vowels, sharp stops (t, p, k), minimal ornamentation. Key: Utilitarian clarity optimized for speed.",
    "Voice: Airy, bright, slightly breathy. Tone: Playful, curious, inviting. Pacing: Fluid, natural rises/falls. Pronunciation: Clear but softened consonants, open vowels, light articulation. Key: Clarity with an approachable, effortless feel."
]
ATTEMPTS_PER_INSTRUCTION = 5  # Вариантов на инструкцию
ATTEMPTS_PER_VOICE = 3  # Дополнительных вариантов на голос
REQUEST_DELAY = 3  # Задержка между запросами (сек)
MAX_RETRIES = 10  # Макс. попыток при ошибках
RETRY_DELAY = 30  # Задержка при повторе (сек)
OUTPUT_DIR = "synthetic_dataset"  # Папка для результатов

# ====================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ====================
def safe_filename(text: str, max_len: int = 10) -> str:
    """Создает безопасное имя файла из текста, заменяя опасные символы на латинские буквы"""
    # Карта замен опасных символов на латинские буквы
    replace_map = {
        '*': 'a',
        '<': 'b',
        '>': 'c',
        ':': 'd',
        '"': 'e',
        '/': 'f',
        '\\': 'g',
        '|': 'h',
        '?': 'i',
        '!': 'j',
        '@': 'k',
        '#': 'l',
        '$': 'm',
        '%': 'n',
        '^': 'o',
        '&': 'p',
        '(': 'q',
        ')': 'r',
        '[': 's',
        ']': 't',
        '{': 'u',
        '}': 'v',
        ';': 'w',
        "'": 'x',
        ',': 'y',
        '`': 'z',
        '~': 'z'
    }
    cleaned = text.lower()
    for bad, repl in replace_map.items():
        cleaned = cleaned.replace(bad, repl)
    cleaned = re.sub(r'[\s-]+', '_', cleaned)
    # Добавляем хеш для уникальности
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"{cleaned[:max_len]}_{text_hash}"

# ====================
# ОСНОВНОЙ СКРИПТ
# ====================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)
base_dir = Path(OUTPUT_DIR)
base_dir.mkdir(parents=True, exist_ok=True)

total_files = len(WAKE_WORDS) * len(VOICES) * (len(INSTRUCTIONS) * ATTEMPTS_PER_INSTRUCTION + ATTEMPTS_PER_VOICE)
processed_files = 0
start_time = time.time()

for wake_word in WAKE_WORDS:
    word_dir = base_dir / safe_filename(wake_word, 15)
    word_dir.mkdir(exist_ok=True)
    
    for voice in VOICES:
        # Генерация по инструкциям
        for instruction in INSTRUCTIONS:
            safe_instr = safe_filename(instruction, 15)
            
            for i in range(ATTEMPTS_PER_INSTRUCTION):
                filename = word_dir / f"{voice}-{safe_instr}-{i}.wav"
                if filename.exists():
                    continue
                
                retries = 0
                while retries <= MAX_RETRIES:
                    try:
                        with client.audio.speech.with_streaming_response.create(
                            model=MODEL_NAME,
                            voice=voice,
                            input=wake_word,
                            instructions=instruction
                        ) as response:
                            response.stream_to_file(filename)
                        
                        processed_files += 1
                        elapsed = time.time() - start_time
                        print(f"Создан: {filename} | Прогресс: {processed_files}/{total_files} | Время: {elapsed:.1f}с")
                        time.sleep(REQUEST_DELAY)
                        break
                    except Exception as e:
                        print(f"Ошибка ({retries}/{MAX_RETRIES}): {e}")
                        retries += 1
                        time.sleep(RETRY_DELAY)
        
        # Дополнительные варианты без специфичных инструкций
        for j in range(ATTEMPTS_PER_VOICE):
            filename = word_dir / f"{voice}-{safe_instr}-{i}.wav"
            if filename.exists():
                continue
            
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    with client.audio.speech.with_streaming_response.create(
                        model=MODEL_NAME,
                        voice=voice,
                        input=wake_word
                    ) as response:
                        response.stream_to_file(filename)
                    
                    processed_files += 1
                    elapsed = time.time() - start_time
                    print(f"Создан: {filename} | Прогресс: {processed_files}/{total_files} | Время: {elapsed:.1f}с")
                    time.sleep(REQUEST_DELAY)
                    break
                except Exception as e:
                    print(f"Ошибка ({retries}/{MAX_RETRIES}): {e}")
                    retries += 1
                    time.sleep(RETRY_DELAY)

print("Генерация завершена!")