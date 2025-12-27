import numpy as np
import matplotlib.pyplot as plt
import time # Import thư viện thời gian để đo hiệu suất

# --- HÀM DFT THỦ CÔNG (O(N^2)) ĐÃ SỬA LỖI ---
def dft_manual(x):
    """
    Tính DFT rời rạc (Discrete Fourier Transform) thủ công.
    """
    N = len(x)
    X = np.zeros(N, dtype=np.complex128) 
    
    for k in range(N):
        for n in range(N):
            twiddle_factor = np.exp(-1j * 2 * np.pi * k * n / N) #tính toán hệ số W_N
            X[k] += x[n] * twiddle_factor
            
    return X
# ------------------------------------------------

# --- 1. Thiết lập Tham số Lấy mẫu ---
Fs = 50.0  # Tần số lấy mẫu (Hz)
T = 4.0    # Thời gian lấy mẫu (giây)
N = int(Fs * T) # Tổng số mẫu
t = np.linspace(0.0, T, N, endpoint=False)

# --- 2. Định nghĩa Tín hiệu ---
f1, A1 = 1.0, 2.0
f2, A2 = 1.5, 3.0
signal_A = (A1 * np.cos(2.0 * np.pi * f1 * t)) + (A2 * np.cos(2.0 * np.pi * f2 * t))

# --- 3. Thực hiện DFT Thủ công và ĐO THỜI GIAN ---

start_time = time.perf_counter() # Bắt đầu đo
dft_output = dft_manual(signal_A)
end_time = time.perf_counter()   # Kết thúc đo

calculation_time = (end_time - start_time) * 1000 # Chuyển sang mili giây (ms)

# --- 4. Chuẩn hóa và Ánh xạ Tần số (Miền Tần số) ---
magnitude = np.abs(dft_output) #chuyển từ số phức sang phổ biên độ
magnitude = magnitude[:N // 2] #chỉ cần giữ lại nửa phổ đầu

magnitude[1:] = 2 * magnitude[1:] # bù lại một nửa năng lượng đã mất do bỏ nửa sau
magnitude = magnitude / N #chuẩn hóa

freq = np.fft.fftfreq(N, 1/Fs)[:N // 2] #tạo trục tần số


# --- 5. Trực quan hóa Kết quả ---

plt.figure(figsize=(14, 8))

# Biểu đồ 1: Sóng Giao động (Miền Thời gian)
plt.subplot(2, 1, 1)
plt.plot(t, signal_A, label=r'$A(t) = 2\cos(2\pi t) + 3\cos(3\pi t)$', color='blue')
plt.title('1. Tín hiệu Tổng hợp A (Miền Thời gian)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True, linestyle='--')
plt.legend()

# Biểu đồ 2: Phân rã Tín hiệu (Miền Tần số)
plt.subplot(2, 1, 2)
plt.plot(freq, magnitude, color='red')
plt.title(f'2. Phổ Tần số (DFT Thủ công). Thời gian: {calculation_time:.3f} ms')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ (Magnitude)')
plt.xlim(0, Fs / 2)
plt.xticks(np.arange(0, Fs / 2 + 0.5, 0.5)) 
plt.grid(True, linestyle='--')

# Thêm chú thích
plt.annotate(f'A1: {A1} @ {f1} Hz', 
             xy=(f1, A1), 
             xytext=(f1 + 0.1, A1 + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

plt.annotate(f'A2: {A2} @ {f2} Hz', 
             xy=(f2, A2), 
             xytext=(f2 + 0.1, A2 + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

plt.tight_layout()
plt.show()

# --- Kết quả in ra Console ---
print("-" * 50)
print(f"Tổng số mẫu N: {N}")
print(f"Thời gian tính toán DFT (O(N^2)): {calculation_time:.3f} ms")
print(f"Biên độ tại f1={f1} Hz: {magnitude[int(f1*T)]:.2f}")
print(f"Biên độ tại f2={f2} Hz: {magnitude[int(f2*T)]:.2f}")