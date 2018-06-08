# coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack


def load_train_data(filenames, args=args):

    train_data = np.empty((0, args.sig_len), np.float32)

    for filename in filenames:
        x, sf = readWave(filename, args.stereo)
        if len(x) < args.sig_len + args.fp:
            continue

        idx = (int)((len(x) - args.sig_len) / args.fp) + 1
        for i in range(idx):
            data = x[i*args.fp:(i+1)*args.fp+args.sig_len]
            train_data = np.append(train_data, np.array([data]), axis=0)

    return train_data


def readWave(filename, stereo=False):
    wf = wave.open(filename, "r")
    sf = wf.getframerate()
    data = wf.readframes(wf.getnframes())
    if wf.getnchannels() == 2:
        x = np.frombuffer(data, dtype="int16")
        left = x[::2]
        right = x[1::2]
        x = 0.5 * (left + right) / (2.0 * 32768.0)  # 範囲を[-0.5:0.5]に正規化
    else:
        x = 0.5 * np.frombuffer(data, dtype="int16") / \
            32768.0  # 範囲を[-0.5:0.5]に正規化
    wf.close()
    if not stereo:
        return x.astype(np.float32), float(sf)
    else:
        left = 0.5 * left / 32768.0
        right = 0.5 * right / 32768.0
        return left.astype(np.float32), right.astype(np.float32), float(sf)


def writeWave(signal, sf, name="write"):

    # [-0.5:0.5]の範囲で振幅があるという前提
    signal = signal * 2.0
    signal = signal.astype(np.int16)
    save_wav = wave.Wave_write(name+".wav")
    save_wav.setnchannels(1)
    save_wav.setsampwidth(2)
    save_wav.setframerate(sf)
    save_wav.writeframes(signal)
    save_wav.close()
