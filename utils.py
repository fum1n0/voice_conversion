# coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack
import csv


def load_train_data(filenames, args):

    for filename in filenames:

        x, fs = readWave(filename, args.stereo)
        if fs != args.fs or len(x) < args.fl + args.fp:
            continue

        y = x.copy()
        y = y[args.fp:]

        id_x = (int)(len(x) / args.fl)
        sub = args.fl * (id_x+1) - len(x)
        z_x = np.zeros(sub)
        x = np.append(x, z_x)

        id_y = (int)(len(y) / args.fl)
        sub = args.fl * (id_y+1) - len(y)
        z_y = np.zeros(sub)
        y = np.append(y, z_y)

        mini_id = min(id_x, id_y) + 1

        wav_x = np.split(x, id_x + 1)
        wav_y = np.split(y, id_y + 1)

        wav_ = np.empty((mini_id*2, args.fl), np.float32)

        wav_[0::2] = wav_x[:mini_id]
        wav_[1::2] = wav_y[:mini_id]

        np.savetxt(filename[:-4]+".csv", wav_, delimiter=",")


def load_test_data(filename, args):

    x, fs = readWave(filename, args.stereo)
    if fs != args.fs or len(x) < args.fl + args.fp:
        return data

    y = x.copy()
    y = y[args.fp:]

    id_x = (int)(len(x) / args.fl)
    sub = args.fl * (id_x+1) - len(x)
    z_x = np.zeros(sub)
    x = np.append(x, z_x)

    id_y = (int)(len(y) / args.fl)
    sub = args.fl * (id_y+1) - len(y)
    z_y = np.zeros(sub)
    y = np.append(y, z_y)

    mini_id = min(id_x, id_y) + 1

    wav_x = np.split(x, id_x + 1)
    wav_y = np.split(y, id_y + 1)

    wav_ = np.empty((mini_id*2, args.fl), np.float32)

    wav_[0::2] = wav_x[:mini_id]
    wav_[1::2] = wav_y[:mini_id]

    np.savetxt(filename[:-4] + ".csv", wav_, delimiter=",")
    
    return wav_


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
