# coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack
import csv


def load_train_data(filenames, args):

    data = np.empty((0, args.fl), np.float32)

    for filename in filenames:
        wav_ = np.empty((0, args.fl), np.float32)
        x, fs = readWave(filename, args.stereo)
        if fs != args.fs or len(x) < args.fl + args.fp:
            continue

        period = (int)(len(x) / args.fp)
        sub = args.fl - (len(x) - args.fp * period)
        for i in range(0, sub):
            x = np.append(x, 0.0)

        idx = (int)((len(x) - args.fl) / args.fp)
        for i in range(idx):
            signal = x[i * args.fp:i * args.fp + args.fl]
            wav_ = np.append(wav_, np.array([signal]), axis=0)
            data = np.append(data, np.array([signal]), axis=0)

        np.savetxt(filename[:-4]+".csv", wav_, delimiter=",")



def load_test_data(filename, args):

    data = np.empty((0, args.fl), np.float32)

    x, fs = readWave(filename, args.stereo)
    if fs != args.fs or len(x) < args.fl + args.fp:
        return data

    period = (int)(len(x) / args.fp)
    sub = args.fl - (len(x) - args.fp * period)
    for i in range(0, sub):
        x = np.append(x, 0.0)

    idx = (int)((len(x) - args.fl) / args.fp)
    for i in range(idx):
        signal = x[i*args.fp:i*args.fp+args.fl]
        data = np.append(data, np.array([signal]), axis=0)

    np.savetxt(filename[:-4]+".csv", data, delimiter=",")
    return data


def load_train_csv(filenames, args):

    data = np.empty((0, args.fl), np.float32)

    for filename in filenames:
        x = np.loadtxt(filename, delimiter=',')
        data = np.append(data, np.array(x), axis=0)

    return data


def load_test_csv(filename, args):

    data = np.empty((0, args.fl), np.float32)

    x = np.loadtxt(filename, delimiter=',')
    data = np.append(data, np.array(x), axis=0)

    return data


def readWave(filename, stereo=False):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    data = wf.readframes(wf.getnframes())
    if wf.getnchannels() == 2:
        x = np.frombuffer(data, dtype="int16") / 32768.0
        left = x[::2]
        right = x[1::2]
        x = (left + right) / 2.0  # 範囲を[-1.0:1.0]に正規化
    else:
        x = np.frombuffer(data, dtype="int16") / \
            32768.0  # 範囲を[-1.0:1.0]に正規化

    wf.close()

    if not stereo:
        return x.astype(np.float32), float(fs)
    else:
        left = left
        right = right

        return left.astype(np.float32), right.astype(np.float32), float(fs)


def writeWave(signal, fs, name="write"):

    mx = max(max(signal), abs(min(signal)))

    if mx < 1.0:
        signal *= 32768.0
    else:
        signal *= 32768.0 / mx

    signal = signal.astype(np.int16)
    save_wav = wave.Wave_write(name+".wav")
    save_wav.setnchannels(1)
    save_wav.setsampwidth(2)
    save_wav.setframerate(fs)
    save_wav.writeframes(signal)
    save_wav.close()


def cut_signal(signal, fl, fp, c_power):

    period = (int)(len(signal) / fp)

    sub = fl - (len(signal) - fp * period)

    for i in range(0, sub):
        signal = np.append(signal, 0.0)

    period = int((len(signal)-fl) / fp)

    hammingWindow = np.hamming(fl)
    cut_signal = np.zeros(fl + period * fp)

    i = 0
    j = -1
    while i < period:
        frame = signal[i * fp:i * fp + fl]
        frame_filter = hammingWindow * frame * 2.0
        sp = np.fft.fft(frame_filter)
        power = np.square((np.abs(sp)/fl))
        avg_power = np.log10(np.mean(power) + 1e-100)
        if c_power < avg_power:
            j = j + 1
            cut_signal[j * fp:j * fp + fl] = frame

        i = i + 1
    cut_signal = cut_signal[:fl+j*fp]

    return cut_signal
