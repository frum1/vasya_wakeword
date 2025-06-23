import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import torch
import random
from typing import List, Tuple
import torchaudio
import acoustics
from openwakeword.data import read_audio, truncate_clip, mix_clip, get_frame_labels, reverberate, mix_clips_batch

import openwakeword
import openwakeword.data
import openwakeword.utils

import datasets

F = openwakeword.utils.AudioFeatures()

NEGATIVE_DATASETS = [
    "dataset/fma/",
    "dataset/fsd/",
    "dataset/cv"
]

POSITIVE_DATASETS = [
    "dataset/vasya_norm",
    "dataset/positive"
]


def safe_mix_clips_batch(
        foreground_clips: List[str],
        background_clips: List[str],
        combined_size: int,
        labels: List[int] = [],
        batch_size: int = 32,
        snr_low: float = 0,
        snr_high: float = 0,
        start_index: List[int] = [],
        foreground_durations: List[float] = [],
        foreground_truncate_strategy: str = "random",
        rirs: List[str] = [],
        rir_probability: int = 1,
        volume_augmentation: bool = True,
        generated_noise_augmentation: float = 0.0,
        shuffle: bool = True,
        return_sequence_labels: bool = False,
        return_background_clips: bool = False,
        return_background_clips_delay: Tuple[int, int] = (0, 0),
        seed: int = 0
        ):
    """
    Модифицированная версия mix_clips_batch с защитой от выхода за пределы диапазона
    """
    # Set random seed, if needed
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Check and Set start indices, if needed
    if not start_index:
        start_index = [0]*batch_size
    else:
        if min(start_index) < 0:
            raise ValueError("Error! At least one value of the `start_index` argument is <0. Check your inputs.")

    # Make dummy labels
    if not labels:
        labels = [0]*len(foreground_clips)

    if shuffle:
        p = np.random.permutation(len(foreground_clips))
        foreground_clips = np.array(foreground_clips)[p].tolist()
        start_index = np.array(start_index)[p].tolist()
        labels = np.array(labels)[p].tolist()
        if foreground_durations:
            foreground_durations = np.array(foreground_durations)[p].tolist()

    for i in range(0, len(foreground_clips), batch_size):
        # Load foreground clips/start indices and truncate as needed
        sr = 16000
        start_index_batch = start_index[i:i+batch_size]
        foreground_clips_batch = [read_audio(j) for j in foreground_clips[i:i+batch_size]]
        foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]
        if foreground_durations:
            foreground_clips_batch = [truncate_clip(j, int(k*sr), foreground_truncate_strategy)
                                      for j, k in zip(foreground_clips_batch, foreground_durations[i:i+batch_size])]
        labels_batch = np.array(labels[i:i+batch_size])

        # Load background clips and pad/truncate as needed
        background_clips_batch = [read_audio(j) for j in random.sample(background_clips, batch_size)]
        background_clips_batch = [j[0] if len(j.shape) > 1 else j for j in background_clips_batch]
        background_clips_batch_delayed = []
        delay = np.random.randint(return_background_clips_delay[0], return_background_clips_delay[1] + 1)
        for ndx, background_clip in enumerate(background_clips_batch):
            if background_clip.shape[0] < (combined_size + delay):
                repeated = background_clip.repeat(
                    np.ceil((combined_size + delay)/background_clip.shape[0]).astype(np.int32)
                )
                background_clips_batch[ndx] = repeated[0:combined_size]
                background_clips_batch_delayed.append(repeated[0+delay:combined_size + delay].clone())
            elif background_clip.shape[0] > (combined_size + delay):
                r = np.random.randint(0, max(1, background_clip.shape[0] - combined_size - delay))
                background_clips_batch[ndx] = background_clip[r:r + combined_size]
                background_clips_batch_delayed.append(background_clip[r+delay:r + combined_size + delay].clone())

        # Mix clips at snr levels
        snrs_db = np.random.uniform(snr_low, snr_high, batch_size)
        mixed_clips = []
        sequence_labels = []
        for fg, bg, snr, start in zip(foreground_clips_batch, background_clips_batch,
                                      snrs_db, start_index_batch):
            if bg.shape[0] != combined_size:
                raise ValueError(bg.shape)
            mixed_clip = mix_clip(fg, bg, snr, start)
            sequence_labels.append(get_frame_labels(combined_size, start, start+fg.shape[0]))

            if np.random.random() < generated_noise_augmentation:
                noise_color = ["white", "pink", "blue", "brown", "violet"]
                noise_clip = acoustics.generator.noise(combined_size, color=np.random.choice(noise_color))
                noise_clip = torch.from_numpy(noise_clip/noise_clip.max())
                mixed_clip = mix_clip(mixed_clip, noise_clip, np.random.choice(snrs_db), 0)

            mixed_clips.append(mixed_clip)

        mixed_clips_batch = torch.vstack(mixed_clips)
        sequence_labels_batch = torch.from_numpy(np.vstack(sequence_labels))

        # Apply reverberation to the batch (from a single RIR file)
        if rirs:
            if np.random.random() <= rir_probability:
                rir_waveform, sr = torchaudio.load(random.choice(rirs))
                if rir_waveform.shape[0] > 1:
                    rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
                mixed_clips_batch = reverberate(mixed_clips_batch, rir_waveform, rescale_amp="avg")

        # Apply volume augmentation
        if volume_augmentation:
            volume_levels = np.random.uniform(0.02, 1.0, mixed_clips_batch.shape[0])
            mixed_clips_batch = (volume_levels/mixed_clips_batch.max(axis=1)[0])[..., None]*mixed_clips_batch
        else:
            # Normalize clips only if max value is outside of [-1, 1]
            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

        # === КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ ===
        # Добавляем явное ограничение значений перед преобразованием
        mixed_clips_batch = torch.clamp(mixed_clips_batch, -1.0, 1.0)
        
        clean_and_validate_audio(mixed_clips_batch.numpy())
        mixed_clips_batch = (mixed_clips_batch*32767).astype(np.int16)

        # Remove any clips that are silent (happens rarely when mixing/reverberating)
        error_index = np.where(mixed_clips_batch.max(axis=1) != 0)[0]
        mixed_clips_batch = mixed_clips_batch[error_index]
        labels_batch = labels_batch[error_index]
        sequence_labels_batch = sequence_labels_batch[error_index]

        if not return_background_clips:
            yield mixed_clips_batch, labels_batch if not return_sequence_labels else sequence_labels_batch, None
        else:
            # === КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ ===
            background_clips_batch_delayed = torch.vstack(background_clips_batch_delayed)
            background_clips_batch_delayed = torch.clamp(background_clips_batch_delayed, -1.0, 1.0)
            background_clips_batch_delayed = (background_clips_batch_delayed.numpy()*32767).astype(np.int16)[error_index]
            
            yield (mixed_clips_batch,
                   labels_batch if not return_sequence_labels else sequence_labels_batch,
                   background_clips_batch_delayed)


def clean_and_validate_audio(audio_tensor):
    """Очистка и проверка аудио тензора"""
    # Конвертируем в numpy для удобства обработки
    audio_np = audio_tensor.numpy()
    
    # Проверка на NaN/Inf
    if np.isnan(audio_np).any() or np.isinf(audio_np).any():
        print("Обнаружены нечисловые значения в аудио. Заменяю на 0.")
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Проверка диапазона
    min_val = np.min(audio_np)
    max_val = np.max(audio_np)
    if min_val < -1.0 or max_val > 1.0:
        print(f"Аудио выходит за пределы [-1, 1]: min={min_val:.4f}, max={max_val:.4f}")
        audio_np = np.clip(audio_np, -1.0, 1.0)
    
    # Проверка на полную тишину (все нули)
    if np.all(audio_np == 0):
        print("Обнаружено полностью нулевое аудио. Генерирую белый шум.")
        audio_np = np.random.uniform(-0.01, 0.01, audio_np.shape)
    
    return audio_np


negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
    NEGATIVE_DATASETS, # list of directories with negative clips
    min_length_secs = 1.0, # minimum clip length in seconds
    max_length_secs = 60*30, # maximum clip length in seconds
    duration_method = "header" # use the file header to calculate duration
)

print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)//3600} hours")

audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

# Get audio embeddings (features) for negative clips and save to .npy file
# Process files by batch and save to Numpy memory mapped file so that
# an array larger than the available system memory can be created

batch_size = 64 # number of files to load, compute features, and write to mmap at a time
clip_size = 3  # the desired window size (in seconds) for the trained openWakeWord model
N_total = int(sum(negative_durations)//clip_size) # maximum number of rows in mmap file
n_feature_cols = F.get_embedding_shape(clip_size)

output_file = "negative_features.npy"
output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

row_counter = 0
for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
    # Load data in batches and shape into rectangular array
    wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
    wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)
    
    # Compute features (increase ncpu argument for faster processing)
    features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
    
    # Save computed features to mmap array file (stopping once the desired size is reached)
    if row_counter + features.shape[0] > N_total:
        fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
        fp.flush()
        break
    else:
        fp[row_counter:row_counter+features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()
        
# Trip empty rows from the mmapped array
openwakeword.data.trim_mmap(output_file)

# Положительные клипы
positive_clips, durations = openwakeword.data.filter_audio_paths(
    POSITIVE_DATASETS,
    min_length_secs=1.0,
    max_length_secs=2.0,
    duration_method="header"
)
print(f"{len(positive_clips)} positive clips after filtering")

sr = 16000
total_length_seconds = 3
total_length = int(sr * total_length_seconds)

jitters = (np.random.uniform(0, 0.2, len(positive_clips)) * sr).astype(np.int32)
starts = [total_length - (int(np.ceil(i * sr)) + j) for i, j in zip(durations, jitters)]
ends = [int(i * sr) + j for i, j in zip(durations, starts)]

batch_size = 8
mixing_generator = mix_clips_batch(
    foreground_clips=positive_clips,
    background_clips=negative_clips,
    combined_size=total_length,
    batch_size=batch_size,
    snr_low=5,
    snr_high=15,
    start_index=starts,
    volume_augmentation=True,
)

N_total = len(positive_clips)
n_feature_cols = F.get_embedding_shape(total_length_seconds)

output_file = "vasya.npy"
output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])

fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

row_counter = 0
for batch in tqdm(mixing_generator, total=N_total // batch_size):
    batch, lbls, background = batch[0], batch[1], batch[2]
    
    try:
        # Вычисляем признаки для всего батча
        features = F.embed_clips(batch, batch_size=256)
    except Exception as e:
        print(f"Ошибка в батче: {e}. Обработка по одному клипу...")
        valid_features = []
        for clip in batch:
            try:
                # Пробуем обработать каждый клип отдельно
                feat = F.embed_clips([clip], batch_size=1)
                valid_features.append(feat)
            except Exception:
                print("Пропущен битый/некорректный аудиоклип")
                continue
        
        if not valid_features:
            continue  # Пропускаем весь батч, если не удалось обработать ни один клип
        features = np.concatenate(valid_features, axis=0)
    
    # Сохраняем результат
    if row_counter + features.shape[0] > N_total:
        remaining = N_total - row_counter
        fp[row_counter:row_counter + remaining, :, :] = features[:remaining]
    else:
        fp[row_counter:row_counter + features.shape[0], :, :] = features
    
    row_counter += features.shape[0]
    fp.flush()
    
    if row_counter >= N_total:
        break

# Обрезаем пустые строки в файле
openwakeword.data.trim_mmap(output_file)