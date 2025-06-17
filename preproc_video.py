import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_video
import torchaudio.transforms as T
import librosa

import os
import shutil
import json


def video_frame_preproc(video_frame, num_sec):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = video_frame[np.linspace(0, video_frame.size(0)-1, num_sec)].to(torch.float32)
    frames = torch.permute(frames, (0, 3, 1, 2))
    return transform(frames)


def audio_frame_preproc(audio_frame, num_sec, audio_sr, target_audio_sr):
    mono_audio = librosa.to_mono(audio_frame.numpy().astype(np.float32))
    if audio_sr != target_audio_sr:
        resampled_audio = librosa.resample(mono_audio, orig_sr=audio_sr, target_sr=target_audio_sr)

    if resampled_audio.shape[0] < target_audio_sr * num_sec:
        padding_size = target_audio_sr * num_sec - resampled_audio.shape[0]
        resized_audio = np.concat([resampled_audio, np.zeros(padding_size)])
    else:
        resized_audio = resampled_audio[:target_audio_sr * num_sec]

    resized_audio = torch.tensor(resized_audio.reshape((num_sec, -1)))

    mel_spectrogram = T.MelSpectrogram(sample_rate=target_audio_sr,
                                       win_length=25,
                                       hop_length=10,
                                       n_mels=64)
    return mel_spectrogram(resized_audio)


def preproc_file(file_path, num_sec, all_sec, target_audio_sr):
    video_samples = []
    audio_samples = []

    for start in range(0, all_sec, num_sec):
        video_tensor, audio_tensor, info = read_video(file_path, start_pts=start,
                                                      end_pts=start + num_sec, pts_unit='sec')

        video_frames = video_frame_preproc(video_tensor, num_sec)
        audio_frames = audio_frame_preproc(audio_tensor, num_sec, info['audio_fps'], target_audio_sr)

        video_samples.append(video_frames)
        audio_samples.append(audio_frames)

    return torch.cat(video_samples), torch.cat(audio_samples)


def preproc_timestamps(start, end, all_sec):
    start_sec = start[0] * 3600 + start[1] * 60 + start[2]
    end_sec = end[0] * 3600 + end[1] * 60 + end[2]

    if start_sec > all_sec:
        return [0] * all_sec

    end_sec = min(end_sec, all_sec)
    return [0]*start_sec + [1]*(end_sec - start_sec) + [0]*(all_sec - end_sec)


def preproc_data(data_path, new_folder_name, num_sec, all_sec, target_audio_sr):
    dirpath, dirnames, filenames = next(os.walk(data_path))
    video_names = [path for path in filenames if path.endswith('mp4')]
    labels_path = f"{data_path}/labels.json"

    with open(labels_path) as f:
        labels = json.load(f)

    if os.path.exists(f"./{new_folder_name}/"):
        shutil.rmtree(f"./{new_folder_name}/")
    os.mkdir(f"./{new_folder_name}/")

    new_labels = dict()

    for video_name in video_names:
        os.mkdir(f"./{new_folder_name}/{video_name.split('.')[0]}")
        video_tensor, audio_tensor = preproc_file(f"{data_path}/{video_name}", num_sec, all_sec, target_audio_sr)

        torch.save(video_tensor, f"./{new_folder_name}/{video_name.split('.')[0]}/video_tensor.pt")
        torch.save(audio_tensor, f"./{new_folder_name}/{video_name.split('.')[0]}/audio_tensor.pt")

        start_intro = tuple(map(int, labels[video_name.split('.')[0]]['start'].split(':')))
        end_intro = tuple(map(int, labels[video_name.split('.')[0]]['end'].split(':')))

        video_labels = preproc_timestamps(start_intro, end_intro, all_sec)
        new_labels[video_name.split('.')[0]] = video_labels

        #break

    with open(f"./{new_folder_name}/labels.json", 'w') as f:
        json.dump(new_labels, f)


if __name__ == '__main__':
    num_sec = 5
    all_sec = 4 * 60
    target_audio_sr = 16_000

    train_data = './data_test_short'
    new_train_data = './train_data'
    preproc_data(train_data, new_train_data, num_sec, all_sec, target_audio_sr)
