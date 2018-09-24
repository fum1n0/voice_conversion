# coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack
import csv


def load_data(filename, args):

    x, fs = readWave(filename, args.stereo)

    if len(x) < args.fl + args.fp:
        return np.empty((0, args.fl), np.float32)

    if fs != args.fs:
        if fs > args.fs:
            x, fs = upsampling(x, fs, fs / args.fs)
        else:
            x, fs = downsampling(x, fs, args.fs / fs)
        writeWave(x, fs, filename[:-4])

    y = x.copy()
    y = y[args.fp:]

    id_x = (int)(len(x) / args.fl)
    id_y = (int)(len(y) / args.fl)

    mini_id = min(id_x, id_y)

    wav_x = x[:id_x*args.fl].reshape(-1, args.fl)
    wav_y = y[:id_y*args.fl].reshape(-1, args.fl)

    wav_ = np.empty((mini_id*2, args.fl), np.float32)

    wav_[0::2] = wav_x[:mini_id]
    wav_[1::2] = wav_y[:mini_id]

    return wav_


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


def upsampling(data, fs, conversion_rate):
    """
    アップサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    アップサンプリング後のデータとサンプリング周波数を返す．
    """
    # 補間するサンプル数を決める
    interpolationSampleNum = conversion_rate-1

    # FIRフィルタの用意をする
    nyqF = (fs*conversion_rate)/2.0     # 変換後のナイキスト周波数
    cF = (fs/2.0-500.)/nyqF             # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    taps = 511                          # フィルタ係数（奇数じゃないとだめ）
    b = sig.firwin(taps, cF)   # LPFを用意

    # 補間処理
    upData = []
    for d in data:
        upData.append(d)
        # 1サンプルの後に，interpolationSampleNum分だけ0を追加する
        for i in range(interpolationSampleNum):
            upData.append(0.0)

    # フィルタリング
    resultData = sig.lfilter(b, 1, upData)
    return (resultData, (int)(fs*conversion_rate))


def downsampling(data, fs, conversion_rate):
    """
    ダウンサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    アップサンプリング後のデータとサンプリング周波数を返す．
    """
    # 間引くサンプル数を決める
    decimationSampleNum = conversion_rate-1

    # FIRフィルタの用意をする
    nyqF = (fs/conversion_rate)/2.0             # 変換後のナイキスト周波数
    # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    cF = (fs/conversion_rate/2.0-500.)/nyqF
    taps = 511                                  # フィルタ係数（奇数じゃないとだめ）
    b = sig.firwin(taps, cF)           # LPFを用意

    # フィルタリング
    data = sig.lfilter(b, 1, data)

    # 間引き処理
    downData = []
    for i in range(0, len(data), decimationSampleNum+1):
        downData.append(data[i])

    return (downData, (int)(fs/conversion_rate))
