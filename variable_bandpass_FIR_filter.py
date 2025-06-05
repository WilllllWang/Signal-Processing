# Design of variable_bandpass_filter with two variable parameters

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

N = 40
Mp = 5
Mq = 5
wp1 = 0.3 * math.pi
wp2 = 0.45 * math.pi
wp3 = 0.55 * math.pi  ## wp3 > wp2
wp4 = 0.7 * math.pi
wt = 0.2 * math.pi
Nw = 200
Np = 60
Nq = 60
##
##
NH = N // 2
nm = (NH + 1) * (Mp + 1) * (Mq + 1)
deltaw = math.pi / Nw
deltap = 1 / Np
deltaq = 1 / Nq
NV = np.arange(0, NH + 1)
NV = NV[:, np.newaxis]
##
##
r = np.zeros((nm, 1))
Qp = np.zeros((nm, nm))
Qs1 = np.zeros((nm, nm))
Qs2 = np.zeros((nm, nm))
ppass = 0
stop1 = 0
stop2 = 0

for iq in range(0, Nq + 1):
    print("iq: ", iq)
    q = -0.5 + iq * deltaq
    wpu = (q + 0.5) * (wp4 - wp3) + wp3

    for ip in range(0, Np + 1):
        p = -0.5 + ip * deltap
        wpd = (p + 0.5) * (wp2 - wp1) + wp1

        for iw in range(0, Nw + 1):
            w = iw * deltaw

            i = 0
            cwpq = np.zeros((nm, 1))
            for imq in range(0, Mq + 1):
                for imp in range(0, Mp + 1):
                    cwpq[i * (NH + 1):(i + 1) * (NH + 1), 0] = \
                        np.cos(NV[:, 0] * w) * (p ** imp) * (q ** imq)
                    i = i + 1

            if w <= wpd - wt:
                stop1 = stop1 + 1
                Qs1 = Qs1 + cwpq @ np.transpose(cwpq)
            elif w >= wpd and w <= wpu:
                ppass = ppass + 1
                r = r - 2 * cwpq
                Qp = Qp + cwpq @ np.transpose(cwpq)
            elif w >= wpu + wt:
                stop2 = stop2 + 1
                Qs2 = Qs2 + cwpq @ np.transpose(cwpq)

r = r / ppass
Qp = Qp / ppass
Qs1 = Qs1 / stop1
Qs2 = Qs2 / stop2
a = -0.5 * np.linalg.inv(Qp + Qs1 + Qs2) @ r
##
a3 = np.reshape(a, (Mq + 1, Mp + 1, NH + 1))
##
## plot, q = -0.5
##
q = -0.5
a2 = a3[0, :, :]
for imq in range(1, Mq + 1):
    a2 = a2 + a3[imq, :, :] * (q ** imq)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mp + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 1)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mp + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, q = -0.5')
##
## plot, q = 0
##
q = 0
a2 = a3[0, :, :]
for imq in range(1, Mq + 1):
    a2 = a2 + a3[imq, :, :] * (q ** imq)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mp + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 2)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mp + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, q = 0')
##
## plot, q = 0.5
##
q = 0.5
a2 = a3[0, :, :]
for imq in range(1, Mq + 1):
    a2 = a2 + a3[imq, :, :] * (q ** imq)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mp + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 3)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mp + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, q = 0.5')
##
## plot, p = -0.5
##
p = -0.5
a2 = a3[:, 0, :]
for imp in range(1, Mp + 1):
    a2 = a2 + a3[:, imp, :] * (p ** imp)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mq + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 4)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mq + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, p = -0.5')
##
## plot, p = 0
##
p = 0
a2 = a3[:, 0, :]
for imp in range(1, Mp + 1):
    a2 = a2 + a3[:, imp, :] * (p ** imp)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mq + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 5)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mq + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, p = 0')
##
## plot, p = 0.5
##
p = 0.5
a2 = a3[:, 0, :]
for imp in range(1, Mp + 1):
    a2 = a2 + a3[:, imp, :] * (p ** imp)

a2 = np.transpose(a2)
h2 = np.zeros((N + 1, Mq + 1))
h2[NH, :] = a2[0, :]
h2[0:NH, :] = 0.5 * np.flipud(a2[1:NH + 1, :])
h2[NH + 1:N + 1, :] = 0.5 * a2[1:NH + 1, :]
##
plt.subplot(2, 3, 6)
MR = np.zeros((Nw + 1, Np + 1, 1))
for ip in range(0, Np + 1):
    p = -0.5 + ip * deltap
    h = h2[:, 0]
    for im in range(1, Mq + 1):
        h = h + h2[:, im] * p ** im
    rr = np.linspace(0, math.pi, num=Nw + 1)
    rr = rr[:, np.newaxis]
    MRR = np.absolute(signal.freqz(h, 1, rr))
    MR[:, ip] = MRR[1]

for i in range(0, Np + 1):
    plt.plot(rr / math.pi, MR[:, i])
plt.axis([0, 1, 0, 1.1])
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude response')
plt.title('variable bandpass filter, p = 0.5')

plt.show()
