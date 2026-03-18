import torch
import numpy as np
import torchaudio
import librosa
import soundfile as sf
import os
import sys
import scipy.signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 将上级目录添加到 sys.path 中，以便导入 config 模块
import config


# load audio file, support resampling and mono conversion
def load_audio(file_path,target_sr=config.SAMPLE_RATE, mono=True):
    try:
        waveform, sr = torchaudio.load(file_path)
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # 转为单声道
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze().numpy()  # 转为 numpy 数组
        return waveform, target_sr
    except Exception as e:
        print(f"torchaudio 加载失败 ({e})，使用 librosa 加载...")
        waveform, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return waveform, sr



# preemphasis signal, improves high-frequency components for better spectrograms
def preemphasis(signal):
    return scipy.signal.lfilter([1, -config.PREEMPHASIS], [1], signal)

# de-emphasis signal, inverse of preemphasis
def deemphasis(signal):
    return scipy.signal.lfilter([1], [1, -config.PREEMPHASIS], signal)


# convert waveform to mel spectrogram
def waveform_to_mel(waveform, apply_preemphasis=True, n_mels=None):
    # 如果调用时没传 n_mels，就默认用 config 里的
    if n_mels is None:
        n_mels = config.N_MELS

    if apply_preemphasis:    
        waveform = preemphasis(waveform)
    
    # get mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=config.SAMPLE_RATE, 
        n_fft=config.N_FFT, 
        hop_length=config.HOP_SIZE, 
        win_length=config.WIN_LENGTH, 
        n_mels=n_mels, 
        fmin=config.FMIN, 
        fmax=config.FMAX,
        power=1.0  
    )
    
    
    mel_db = 20 * np.log10(np.maximum(1e-5, mel_spec)) 
    mel_norm = (mel_db - config.REF_DB) / (config.MAX_DB - config.REF_DB)
    mel_norm = np.clip(mel_norm, 0, 1)

    return mel_spec, mel_norm


# save waveform to file
def save_waveform(waveform, file_path, sr=config.SAMPLE_RATE):
    sf.write(file_path, waveform, sr)


# test the audio tools
if __name__ == "__main__":
    test_file = os.path.join(config.RAW_DATA_DIR, '001.wav')
    waveform, sr = load_audio(test_file)
    pre_waveform = preemphasis(waveform)
    print(f"---Loaded waveform shape: {waveform.shape}, Sample Rate: {sr}")

    mel_spec, mel_norm = waveform_to_mel(waveform)
    print(f"---Mel Spectrogram shape: {mel_spec.shape} \nNormalized Mel shape: {mel_norm.shape}")

    # save the waveform back to a new file to test saving
    save_path = os.path.join(config.OUTPUT_DIR, 'test_output_sharp.wav')
    save_waveform(pre_waveform, save_path)
    print(f"---Saved test waveform to: {save_path}")
    print("---测试完成！")


    