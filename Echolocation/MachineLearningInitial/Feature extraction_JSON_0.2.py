#!/usr/bin/env python3
import os
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate, coherence

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def load_mono_audio(file_path, sr=None):
    """
    Loads a single-channel (mono) audio file using soundfile.
    Returns (signal, sample_rate).
    If 'sr' is specified and differs from the file, resample.
    """
    data, sample_rate = sf.read(file_path)
    # data may have shape (samples,) if truly mono.
    if data.ndim == 2:
        # If for some reason it has 2 channels, just take the first
        data = data[:, 0]
    if sr is not None and sample_rate != sr:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
        sample_rate = sr
    return data, sample_rate

def combine_stereo(left_signal, right_signal):
    """
    Combine two mono numpy arrays (same length) into a 2D array
    representing a stereo signal: shape = (samples, 2).
    """
    min_len = min(len(left_signal), len(right_signal))
    # Trim to the shorter length if they differ
    left_signal = left_signal[:min_len]
    right_signal = right_signal[:min_len]
    stereo_signal = np.column_stack((left_signal, right_signal))
    return stereo_signal

def compute_time_features(signal, sr, frame_length=2048, hop_length=512):
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
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "mfccs_mean": np.mean(mfccs, axis=1).tolist(),
        "mfccs_std": np.std(mfccs, axis=1).tolist()
    }
    return features

def compute_spatial_features(left_signal, right_signal, sr):
    """
    Compute spatial features that rely on having both channels.
    """
    features = {}
    # ITD using cross-correlation
    corr = correlate(left_signal, right_signal, mode='full')
    lag_idx = np.argmax(np.abs(corr))
    lag = lag_idx - (len(left_signal) - 1)
    itd = lag / sr
    features["itd_seconds"] = float(itd)
    
    # ILD: difference in RMS energy in dB
    rms_left = np.sqrt(np.mean(left_signal**2))
    rms_right = np.sqrt(np.mean(right_signal**2))
    ild = 20 * np.log10(rms_left / (rms_right + 1e-10))
    features["ild_db"] = float(ild)
    
    # Interaural Coherence using scipy's coherence function
    _, coh = coherence(left_signal, right_signal, fs=sr, nperseg=1024)
    features["interaural_coherence_mean"] = float(np.mean(coh))
    
    # Phase Difference: average phase difference over STFT frames
    stft_left = librosa.stft(left_signal, n_fft=2048, hop_length=512)
    stft_right = librosa.stft(right_signal, n_fft=2048, hop_length=512)
    phase_diff = np.angle(stft_left) - np.angle(stft_right)
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    features["phase_diff_mean"] = float(np.mean(phase_diff))
    
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
    return float(rt60) if rt60 is not None else None

def compute_drr(impulse_response, sr):
    peak_idx = np.argmax(np.abs(impulse_response))
    window_size = int(0.005 * sr)
    direct_energy = np.sum(impulse_response[peak_idx:peak_idx+window_size]**2)
    reverberant_energy = np.sum(impulse_response[peak_idx+window_size:]**2)
    if reverberant_energy > 0:
        drr = 10 * np.log10(direct_energy / reverberant_energy)
        return float(drr)
    else:
        return None

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
    return (float(C50) if C50 is not None else None,
            float(C80) if C80 is not None else None)

def compute_room_acoustics_features(impulse_response, sr):
    features = {}
    rt60 = compute_rt60(impulse_response, sr)
    drr = compute_drr(impulse_response, sr)
    C50, C80 = compute_clarity(impulse_response, sr)
    edt = rt60 / 6 if rt60 is not None else None
    features["rt60_seconds"] = rt60
    features["drr_db"] = drr
    features["edt_seconds"] = float(edt) if edt is not None else None
    features["clarity_C50_db"] = C50
    features["clarity_C80_db"] = C80
    return features

# --------------------------------------------------
# Main Feature Extraction Loop
# --------------------------------------------------

def main():
    """
    This script loops over each subfolder in `dataset_2`, looking for:
      * <ID>_sound_left.wav
      * <ID>_sound_right.wav
    If both are found, we treat them as a stereo pair for spatial features.
    If only one is found, we'll do time & frequency features for that channel but skip spatial.
    """
    dataset_root = "dataset_2"  # Adjust if needed
    output_folder = "Extracted features"
    os.makedirs(output_folder, exist_ok=True)
    
    all_features = {}
    
    folder_number = 0

    # Loop over each subfolder in dataset_2
    for folder_name in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder_name)
        if not os.path.isdir(folder_path):
            continue  # skip non-directories

        folder_number += 1
        renamed_folder = f"{folder_name}_{folder_number}"

        print(f"Processing folder: {folder_path}")
        
        # Attempt to find the left and/or right WAV files
        left_wav_path = None
        right_wav_path = None
        
        for f in os.listdir(folder_path):
            if f.endswith("sound_left.wav"):
                left_wav_path = os.path.join(folder_path, f)
            elif f.endswith("sound_right.wav"):
                right_wav_path = os.path.join(folder_path, f)
        
        # If we didn't find any WAV, skip
        if not left_wav_path and not right_wav_path:
            print(f"  No .wav file found in {folder_name}, skipping.")
            continue
        
        # Load signals
        if left_wav_path and right_wav_path:
            # Both channels exist -> treat them as a stereo pair
            try:
                left_signal, sr_left = load_mono_audio(left_wav_path, sr=None)
                right_signal, sr_right = load_mono_audio(right_wav_path, sr=None)
                
                if sr_left != sr_right:
                    print(f"  Warning: sampling rates differ. Forcing them to match via resample.")
                    # Example: resample right to sr_left
                    right_signal = librosa.resample(right_signal, orig_sr=sr_right, target_sr=sr_left)
                    sr = sr_left
                else:
                    sr = sr_left
                
                # Combine to create a 2D stereo array
                stereo_signal = combine_stereo(left_signal, right_signal)
                
                # Now 'stereo_signal' has shape (samples, 2)
                # We'll extract left & right from it as needed:
                left_ch = stereo_signal[:, 0]
                right_ch = stereo_signal[:, 1]
                
                # Time & frequency features for left channel
                time_features_left = compute_time_features(left_ch, sr)
                freq_features_left = compute_frequency_features(left_ch, sr)
                
                # Time & frequency features for right channel
                time_features_right = compute_time_features(right_ch, sr)
                freq_features_right = compute_frequency_features(right_ch, sr)
                
                # Spatial features
                spatial_features = compute_spatial_features(left_ch, right_ch, sr)
                
                # Check if short enough to compute room acoustics
                duration = len(left_ch) / sr
                room_features = {}
                if duration < 2.0:
                    room_features = compute_room_acoustics_features(left_ch, sr)  # or do some alternate approach
                
                # Pack everything
                folder_features = {
                    "left_channel": {
                        "time_features": time_features_left,
                        "frequency_features": freq_features_left,
                    },
                    "right_channel": {
                        "time_features": time_features_right,
                        "frequency_features": freq_features_right,
                    },
                    "spatial_features": spatial_features,
                    "room_acoustics_features": room_features,
                }
                # Use the folder name as the key in our dictionary
                all_features[renamed_folder] = folder_features
            
            except Exception as e:
                print(f"  Error processing stereo in {folder_name}: {e}")
                continue
        
        else:
            # Only one channel found (left or right). We'll do single-channel extraction.
            # We'll skip spatial features because they require 2 channels.
            single_wav = left_wav_path if left_wav_path else right_wav_path
            try:
                signal, sr = load_mono_audio(single_wav, sr=None)
                
                time_features_single = compute_time_features(signal, sr)
                freq_features_single = compute_frequency_features(signal, sr)
                
                duration = len(signal) / sr
                room_features = {}
                if duration < 2.0:
                    room_features = compute_room_acoustics_features(signal, sr)
                
                folder_features = {
                    "single_channel": {
                        "time_features": time_features_single,
                        "frequency_features": freq_features_single,
                    },
                    "spatial_features": {},  # none
                    "room_acoustics_features": room_features
                }
                all_features[renamed_folder] = folder_features
            
            except Exception as e:
                print(f"  Error processing single channel in {folder_name}: {e}")
                continue
    
    # Save to JSON
    output_file = os.path.join(output_folder, "features_all.json")
    with open(output_file, "w") as f:
        json.dump(all_features, f, indent=4, default=lambda x: float(x))
    
    print(f"\nExtraction complete. Features saved to: {output_file}")

if __name__ == '__main__':
    main()
