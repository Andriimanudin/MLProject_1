import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt

tabel = pd.read_csv('/home/nstrtm/Documents/bezdekiris.csv', header=None, nrows=100)
tabel[4] = tabel[4].str.replace('Iris-setosa','1')
tabel[4] = tabel[4].str.replace('Iris-versicolor','0')

tabel = tabel.astype('float64')

#Inisiasi
alpha = 0.1

teta = np.array([0.1,0.15,0.2,0.3])

pteta= pd.DataFrame(data=teta)

global bias
bias = 0.4

d_bias = 0

h = 0

array_dteta = np.empty(4)

array_x = np.array(tabel.iloc[0,:4])

fakta = tabel.iloc[0,4]

jumlah_error_semua = np.zeros(shape=(60,1))


# by def
def h(x, teta, b, a):
    return np.dot(x.iloc[a, :4], np.transpose(teta) + bias)


def sigmoid(h):
    return 1 / (1 + mt.exp(h))


def error(sigmoid, a):
    return (sigmoid - tabel.iloc[a, 4]) ** 2


def prediksi(sigmoid):
    if sigmoid <= 0.5:
        prediksi = 1
        return prediksi
    else:
        prediksi = 0
        return prediksi


def d_teta(sigmoid, fakta, array_x, i):
    return 2 * (sigmoid - fakta) * (1 - sigmoid) * sigmoid * array_x[i]


def d_bias(sigmoid, fakta):
    return 2 * (sigmoid - fakta) * (1 - sigmoid) * sigmoid


def teta_baru(teta, alpha, d_teta, i):
    return teta[i] - (alpha * d_teta[i])


def bias_baru(bias, alpha, d_bias):
    return bias - (alpha * d_bias)


def tukar_bias(bias, alpha, d_bias_rumus):
    bias = bias_baru(bias, alpha, d_bias_rumus)


def jumlah_error_epoch(error):
    jumlah = jumlah + error


def restart_jumlah(i):
    jumlah_error_semua[i] = jumlah.copy


def epoch():
    global jumlah
    jumlah = 0
    diulang = 60
    for m in range(diulang):

        for i in tabel:
            h_rumus = h(tabel, teta, bias, i).astype('float64')
            sigmoid_rumus = sigmoid(h_rumus)
            error_rumus = error(sigmoid_rumus, i)
            jumlah += error_rumus
            prediksi_rumus = prediksi(sigmoid_rumus)
            for j in range(len(array_x)):
                array_dteta[j] = d_teta(sigmoid_rumus, fakta, array_x, j)
            d_bias_rumus = d_bias(sigmoid_rumus, fakta)
            for j in range(len(teta)):
                teta[j] = teta_baru(teta, alpha, array_dteta, j)
            tukar_bias(bias, alpha, d_bias_rumus)
        jumlah_error_semua[m] = jumlah
        jumlah = 0
        print(jumlah_error_semua[m])
    plt.plot(jumlah_error_semua)
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.show()

epoch()