import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile
import datasets
import io
import soundfile as sf
from datasets.utils.logging import disable_progress_bar

# Отключаем встроенный прогресс-бар datasets
disable_progress_bar()

# Создаем кастомный Features объект, точно соответствующий реальной структуре
features = datasets.Features({
    "client_id": datasets.Value("string"),
    "path": datasets.Value("string"),
    "sentence_id": datasets.Value("string"),
    "sentence": datasets.Value("string"),
    "sentence_domain": datasets.Value("string"),
    "up_votes": datasets.Value("string"),
    "down_votes": datasets.Value("string"),
    "age": datasets.Value("string"),
    "gender": datasets.Value("string"),
    "variant": datasets.Value("string"),
    "locale": datasets.Value("string"),
    "segment": datasets.Value("string"),
    "accent": datasets.Value("string"),
    "audio": {
        "bytes": datasets.Value("binary"),
        "path": datasets.Value("string")
    }
})

# Загружаем с явным указанием структуры
cv_17 = datasets.load_dataset(
    "mozilla-foundation/common_voice_17_0", 
    "ru", 
    split="test", 
    streaming=True,
    features=features  # Явно указываем реальную структуру
)

limit = 5000
count = 0

for example in tqdm(cv_17, total=limit):
    if count >= limit:
        break
        
    try:
        # Извлекаем имя файла из пути
        file_name = example["path"].split("/")[-1].replace(".mp3", ".wav")
        output_path = os.path.join("dataset/cv", file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Прямой доступ к бинарным данным
        audio_bytes = example["audio"]["bytes"]
        
        # Пропускаем пустые файлы
        if not audio_bytes or len(audio_bytes) == 0:
            print(f"Пустой аудиофайл: {example['path']}")
            continue
            
        # Читаем аудио напрямую через soundfile
        with io.BytesIO(audio_bytes) as audio_stream:
            try:
                # Пытаемся прочитать как MP3
                data, samplerate = sf.read(audio_stream)
            except sf.LibsndfileError:
                # Если не MP3, пробуем как WAV
                audio_stream.seek(0)
                data, samplerate = sf.read(audio_stream, format="WAV")
            
            # Конвертируем в моно при необходимости
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            # Ресемплинг до 16kHz
            if samplerate != 16000:
                import resampy
                data = resampy.resample(data, samplerate, 16000, filter='kaiser_best')
                samplerate = 16000
                
            # Конвертируем в 16-bit PCM
            data = (data * 32767).astype(np.int16)
            
            # Сохраняем WAV
            scipy.io.wavfile.write(output_path, samplerate, data)
        
        count += 1
        
    except Exception as e:
        print(f"Ошибка обработки файла {example.get('path', 'unknown')}: {str(e)}")
        continue

print(f"Успешно обработано {count} файлов из {limit}")