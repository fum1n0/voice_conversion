# coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack


def autocorr(x, nlags=None):
    """自己相関関数を求める
    x:     信号
    nlags: 自己相関関数のサイズ（lag=0からnlags-1まで）
           引数がなければ（lag=0からlen(x)-1まですべて）
    """
    N = len(x)
    if nlags == None:
        nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]

    return r


def autocorr_fft(x):
    N = len(x)
    for i in range(0, N):
        x = np.append(x, 0)
    sp = np.fft.fft(x)
    sp = abs(sp)
    sp = sp*sp
    r = np.fft.ifft(sp)

    return r.real


def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める"""
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    if r[0] == 0:
        return a, e[-1]

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]


def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)


def errorSignal(signal, a):
    e = np.zeros(len(signal))
    for i in range(0, len(signal)):
        tmp = 0.0
        for j in range(0, len(a)):
            if j <= i:
                tmp += a[j] * signal[i-j]
        e[i] = tmp

    return e


def wavread(filename, stereo=False):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    data = wf.readframes(wf.getnframes())
    if wf.getnchannels() == 2:
        x = np.frombuffer(data, dtype="int16") / 32768.0  # (-1, 1)に正規化
        left = x[::2]
        right = x[1::2]
        x = (left + right)/2.0
    else:
        x = np.frombuffer(data, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    if not stereo:
        return x, float(fs)
    else:
        return left, right, float(fs)


def writeWave(signal, sf, name="write"):

    mx = max(max(signal), abs(min(signal)))

    if 1.0 < mx:
        signal *= 32768.0/mx
    else:
        signal *= 32768.0

    signal = signal.astype(np.int16)
    save_wav = wave.Wave_write(name+".wav")
    save_wav.setnchannels(1)
    save_wav.setsampwidth(2)
    save_wav.setframerate(sf)
    save_wav.writeframes(signal)
    save_wav.close()


def calcPitchFreq(r_error, fs):
    return fs / (np.argmax(r_error[1:]) + 1)


def createPulse(freq, fs, len):

    pulse = np.zeros(len)
    num = 0
    while num < len:
        pulse[int(num)] = 1.0
        num += fs/freq

    return pulse
