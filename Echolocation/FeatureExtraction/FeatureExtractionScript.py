#!/usr/bin/env python3
import os
import csv
import json
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from scipy.signal import correlate, coherence

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def compute_time_features(signal, sr, frame_length=2048, hop_length=512):
    # rms and zcr are calculated properly but may have to adjust frame length and hop length
    # the default values are 2048 and 512 if you were to not input anything in the function which could maybe be used as an argument
    # if we dont want to change them (?)
    # only looking at one value of rms and zcr rn, maybe consider more (?)
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    features = {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }
    return features

def compute_frequency_features(signal, sr, n_fft=2048, hop_length=512, n_mfcc=13):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    #centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    #mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = {
        #"spectral_centroid_mean": float(np.mean(centroid)),
        #"spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        #"mfccs_mean": np.mean(mfccs, axis=1).tolist(),
        #"mfccs_std": np.std(mfccs, axis=1).tolist()
    }
    return features

def compute_spatial_features(left_signal, right_signal, sr):
    features = {}
    # ITD using cross-correlation
    corr = correlate(left_signal, right_signal, mode='full')
    lag_idx = np.argmax(np.abs(corr))
    lag = lag_idx - (len(left_signal) - 1)
    itd = lag / sr + 1e-10  # Avoid division by zero
    features["itd_seconds"] = float(itd)

    # ILD: difference in RMS energy in dB
    rms_left = np.sqrt(np.mean(left_signal**2))
    rms_right = np.sqrt(np.mean(right_signal**2))
    ild = 20 * np.log10(rms_left / (rms_right + 1e-10))
    features["ild_db"] = float(ild)

    # Interaural Coherence using scipy's coherence function
    _, coh = coherence(left_signal, right_signal, fs=sr, nperseg=1024)
    features["interaural_coherence_mean"] = float(np.mean(coh))

    ## Phase Difference: average phase difference over STFT frames
    #stft_left = librosa.stft(left_signal, n_fft=2048, hop_length=512)
    #stft_right = librosa.stft(right_signal, n_fft=2048, hop_length=512)
    #phase_diff = np.angle(stft_left) - np.angle(stft_right)
    #phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    #features["phase_diff_mean"] = float(np.mean(phase_diff))
    
    return features

def compute_rt60(impulse_response, sr):
    energy = impulse_response**2
    energy_rev = np.cumsum(energy[::-1])[::-1]
    energy_rev_db = 10 * np.log10(energy_rev / np.max(energy_rev) + 1e-10)
    idx = np.where((energy_rev_db <= 0) & (energy_rev_db >= -10))[0]
    if len(idx) > 1:
        x = np.arange(len(idx))
        y = energy_rev_db[idx]
        slope, _ = np.polyfit(x, y, 1)
        rt60 = -60 / slope if slope != 0 else None
    else:
        rt60 = None
    return float(rt60) if rt60 is not None else 1e-10

def compute_drr(impulse_response, sr):
    # looks like it is calculated correctly but there might be issues with the direct part
    # where it is maybe not fully exact, as the first peak is not necessarily where the direct part starts 
    peak_idx = np.argmax(np.abs(impulse_response))
    window_size = int(0.005 * sr)
    direct_energy = np.sum(impulse_response[peak_idx:peak_idx+window_size]**2)
    reverberant_energy = np.sum(impulse_response[peak_idx+window_size:]**2)
    if reverberant_energy > 0:
        drr = 10 * np.log10(direct_energy / reverberant_energy)
        return float(drr)
    else:
        return 1e-10

def compute_clarity(impulse_response, sr):
    peak_idx = np.argmax(np.abs(impulse_response))
    total_energy = np.sum(impulse_response[peak_idx:]**2)
    C50_window = int(0.05 * sr)
    C80_window = int(0.08 * sr)
    early_energy_50 = np.sum(impulse_response[peak_idx:peak_idx+C50_window]**2)
    early_energy_80 = np.sum(impulse_response[peak_idx:peak_idx+C80_window]**2)
    if total_energy > early_energy_50:
        C50 = 10 * np.log10(early_energy_50 / (total_energy - early_energy_50 + 1e-10))
    else:
        C50 = None
    if total_energy > early_energy_80:
        C80 = 10 * np.log10(early_energy_80 / (total_energy - early_energy_80 + 1e-10))
    else:
        C80 = None
    return (float(C50) if C50 is not None else 1e-10,
            float(C80) if C80 is not None else 1e-10)

def compute_room_acoustics_features(impulse_response, sr):
    features = {}
    rt60 = compute_rt60(impulse_response, sr)
    drr = compute_drr(impulse_response, sr)
    C50, C80 = compute_clarity(impulse_response, sr)
    edt = rt60 / 6 if rt60 is not None else None
    features["rt60_seconds"] = rt60
    features["drr_db"] = drr
    features["edt_seconds"] = float(edt) if edt is not None else 1e-10
    features["clarity_C50_db"] = C50
    features["clarity_C80_db"] = C80
    return features


def extract_features(dataset_root_directory, chosen_dataset):
    """
    Extracts features from audio files in the specified dataset directory.
    The dataset should contain subdirectories with audio files named "sound.wav".
    The extracted features are saved in a CSV file.
    The output CSV file will be saved in the "ExtractedFeatures" folder.
    """
    output_folder = "./Echolocation/FeatureExtraction/ExtractedFeatures/" + chosen_dataset
    os.makedirs(output_folder, exist_ok=True)

    records = []
    #skip_ids = {"1", "10", "29", "50", "53", "69", "97", "114", "128", "129", "149", "157", "181", "199", "200", "234", "250",
    #            "263", "283", "396", "441", "465", "472", "477", "502", "522", "527", "538", "645", "668", "686", "697",
    #            "713", "876"}  # Add more as needed

    #skip_ids = {"188", "203" ,"1196", "1214", "1248"}
    
    for folder_name in os.listdir(dataset_root_directory):
        folder_id = folder_name.split("_")[1]
        #if folder_id in skip_ids:
        #    print(f"Skipping folder: {folder_name}")
        #    continue

        folder_path = os.path.join(dataset_root_directory, folder_name)
        if not os.path.isdir(folder_path):
            continue

        #print("Extracting features for data in folder: ", folder_path)

        wav_file_path = None

        for sound_file in os.listdir(folder_path):
            if sound_file.endswith("sound.wav"):
                wav_file_path = os.path.join(folder_path, sound_file)
                break

        if not wav_file_path:
            print("No .wav file found in ", folder_name, ", skipping.")
            continue

        folder_features = {"filename": folder_name}

        try:
            if wav_file_path:
                stereo_signal, sr = sf.read(wav_file_path)

                if stereo_signal.ndim != 2 and stereo_signal.shape[1] < 2:
                    raise ValueError("Input audio file is not stereo.")

                left_ch, right_ch = stereo_signal[:, 0], stereo_signal[:, 1]

                folder_features.update({
                    **{f"left_{k}": v for k, v in compute_time_features(left_ch, sr).items()},
                    **{f"left_{k}": v for k, v in compute_frequency_features(left_ch, sr).items()},
                    **{f"right_{k}": v for k, v in compute_time_features(right_ch, sr).items()},
                    **{f"right_{k}": v for k, v in compute_frequency_features(right_ch, sr).items()},
                    **compute_spatial_features(left_ch, right_ch, sr)
                })

                stereo_signal_duration = len(stereo_signal) / sr
                if stereo_signal_duration < 2.0:
                    folder_features.update(compute_room_acoustics_features(left_ch, sr))

                records.append(folder_features)

        except Exception as e:
            print("Error processing ", folder_name, ": ", e)
            continue
        except RuntimeWarning as r:
            print("Error processing ", folder_name, ": ", r)
            continue

    df = pd.json_normalize(records, sep="_")

    print("DataFrame preview:")
    print(df.head())

    description = df.describe()
    print("DataFrame description:")
    print(description)

    description_file = os.path.join(output_folder, "features_description_CSV.txt")
    description.to_csv(description_file, sep="\t")
    print(f"Description saved to {description_file}")

    output_csv = os.path.join(output_folder, "features_all.csv")
    df.to_csv(output_csv, index=False)
    print(f"DataFrame saved to {output_csv}")
