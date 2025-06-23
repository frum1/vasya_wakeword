import openwakeword
import numpy as np
import pyaudio
import time

# Инициализация модели с вашей кастомной моделью
oww = openwakeword.Model(
    wakeword_model_paths=["vasya.onnx"],
    enable_speex_noise_suppression=True,
    vad_threshold=0.5
)

# Параметры аудиопотока
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # 1280 samples at 16kHz is 80ms

audio = pyaudio.PyAudio()

# Открытие потока
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Переменные для отслеживания состояния
last_activation = time.time()
cooldown = 1.0  # секунды между активациями

try:
    while True:
        # Чтение аудио данных
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        
        # Получение предсказаний
        prediction = oww.predict(audio_data)
        
        # Проверка активации
        if prediction['vasya'] > 0.8:
            current_time = time.time()
            if current_time - last_activation > cooldown:
                last_activation = current_time
                confidence = prediction['vasya']
                print(f"\nАКТИВАЦИЯ! Уверенность: {confidence:.2f}")
                print("Команда распознана!")
                # Здесь можно добавить выполнение действия
                
        # Вывод прогресс-бара
        print(".", end="", flush=True)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nТест остановлен")
    stream.stop_stream()
    stream.close()
    audio.terminate()