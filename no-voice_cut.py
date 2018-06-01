# coding:utf-8
import wave
import numpy as np
import os
import scipy.io.wavfile
import scipy.signal as sig
import scipy.fftpack
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', dest='path',
                    default='./voice.wav', help='path of wav file')
parser.add_argument('--frame_len', dest='fl', type=int,
                    default=1024, help='fft frame length')
parser.add_argument('--frame_period', dest='fl', type=int,
                    default=256, help='fft frame period')
parser.add_argument('--cut_power', dest='cut', type=float,
                    default=-70.0, help='fft frame length')
args = parser.parse_args()


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


if __name__ == '__main__':

    if not os.path.exists(args.path):
        print("not wav path !")
        os.sys.exit()

    signal, fs = wavread(args.path)

    fl = args.fl
    fp = args.fl

    period = int(len(signal) / fp)
    sub = fp - (len(signal) - fp * period)

    for i in range(0, sub):
        signal = np.append(signal, 0.0)
    period = int(len(signal) / fp)

    hammingWindow = np.hamming(fp)
    cut_signal = np.zeros(period * fp)

    i = 0
    j = 0
    while i < period:
        frame = signal[i * fp:i * fp + fl]
        frame_filter = hammingWindow * frame * 2.0
        sp = np.fft.fft(frame_filter)
        power = np.square((np.abs(sp)/fl))
        avg_power = np.log10(np.mean(power) + 1e-100)
        if args.cut < avg_power:
            cut_signal[j * fp:j * fp + fl] = frame
            j = j + 1
        i = i + 1
    cut_signal = cut_signal[:j*fp]
    writeWave(cut_signal, fs, 'cut_{}'.format(os.path.basename(args.path)))
