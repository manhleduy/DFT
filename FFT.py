import numpy as np
import matplotlib.pyplot as plt
import time

# --- HÀM FFT CƠ SỐ 2 CHIA THEO MIỀN THỜI GIAN (O(N log N)) ---
def fft_radix2_dit(x):
    """
    Tính FFT Radix-2 DIT thủ công.
    Độ phức tạp tính toán: O(N log N)
    """
    N = len(x)
    # FFT yêu cầu N là lũy thừa của 2
    if N & (N - 1) != 0:
        raise ValueError("FFT Radix-2 yêu cầu N là lũy thừa của 2.")
        
    M = int(np.log2(N))
    
    # 1. Sắp xếp lại Dữ liệu Đầu vào theo thứ tự Đảo Bit
    X = np.array(x, dtype=np.complex128)
    
    # Hàm đảo bit index
    def bit_reversal(i, M):
        i_rev = 0
        for _ in range(M):
            i_rev = (i_rev << 1) | (i & 1)
            i = i >> 1
        return i_rev

    for i in range(N):
        j = bit_reversal(i, M)
        if j > i:
            X[i], X[j] = X[j], X[i]

    # 2. Thực hiện các Giai đoạn Cánh Bướm (Butterfly Stages)
    # Loop qua các giai đoạn (stages) m = 1 đến M
    for m in range(1, M + 1):
        L = 2**m      # Kích thước khối (block size) hiện tại
        half_L = L // 2 # Khoảng cách butterfly
        
        # Hệ số xoay cơ sở cho giai đoạn này (W_L^1)
        W_L = np.exp(-2j * np.pi / L) 

        # Lặp qua các khối
        for i_block in range(0, N, L):
            W_k = 1.0 # Khởi tạo hệ số xoay W_L^k (k=0)
            
            # Lặp qua các cánh bướm trong khối
            for k in range(half_L):
                i = i_block + k
                
                # Hai đầu vào của butterfly
                A = X[i]
                B = X[i + half_L]
                
                # Thực hiện phép tính cánh bướm DIT:
                temp = W_k * B 
                
                # Đầu ra (in-place)
                X[i] = A + temp      # A' = A + W*B
                X[i + half_L] = A - temp # B' = A - W*B
                
                # Cập nhật hệ số xoay cho cánh bướm tiếp theo
                W_k *= W_L
    
    return X
# ------------------------------------------------------------------


# --- 1. Thiết lập Tham số Lấy mẫu MỚI ---
Fs = 50.0   # Tần số lấy mẫu (Hz)
N = 256     # Tổng số mẫu (Lũy thừa của 2)
T = N / Fs  # Thời gian lấy mẫu (5.12 giây)
t = np.linspace(0.0, T, N, endpoint=False) # Tần số lấy mẫu

# --- 2. Định nghĩa Tín hiệu (Đã Điều chỉnh Tần số) ---
# Đã điều chỉnh để tần số rơi chính xác vào bin FFT (Tránh rò rỉ phổ)
df = Fs / N # Độ phân giải tần số
f1, A1 = 5 * df, 2.0  # f1 ~ 0.976 Hz
f2, A2 = 8 * df, 3.0  # f2 = 1.5625 Hz

# Tín hiệu tổng hợp A
signal_A = (A1 * np.cos(2.0 * np.pi * f1 * t)) + (A2 * np.cos(2.0 * np.pi * f2 * t))

# --- 3. Thực hiện FFT và ĐO THỜI GIAN ---

start_time = time.perf_counter() # Bắt đầu đo
fft_output = fft_radix2_dit(signal_A)
end_time = time.perf_counter()   # Kết thúc đo

calculation_time = (end_time - start_time) * 1000 # Chuyển sang mili giây (ms)

# --- 4. Chuẩn hóa và Ánh xạ Tần số ---
magnitude = np.abs(fft_output)
magnitude = magnitude[:N // 2]  

magnitude[1:] = 2 * magnitude[1:]
magnitude = magnitude / N 

freq = np.fft.fftfreq(N, 1/Fs)[:N // 2]


# --- 5. Trực quan hóa Kết quả ---

plt.figure(figsize=(14, 8))

# Biểu đồ 1: Sóng Giao động (Miền Thời gian)
plt.subplot(2, 1, 1)
plt.plot(t, signal_A, label=f'$A(t) = {A1}\cos(2\pi f_1 t) + {A2}\cos(2\pi f_2 t)$', color='blue')
plt.title('1. Tín hiệu Tổng hợp A (Miền Thời gian)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True, linestyle='--')
plt.legend()

# Biểu đồ 2: Phân rã Tín hiệu (Miền Tần số)
plt.subplot(2, 1, 2)
plt.plot(freq, magnitude, color='red')
plt.title(f'2. Phổ Tần số (FFT Radix-2 DIT). Thời gian: {calculation_time:.3f} ms')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ (Magnitude)')
plt.xlim(0, Fs / 2)
plt.xticks(np.arange(0, Fs / 2 + df, df * 4), rotation=45) # Đặt các mốc tần số rõ ràng
plt.grid(True, linestyle='--')

# Thêm chú thích
plt.annotate(f'A1: {A1} @ {f1:.3f} Hz', 
             xy=(f1, A1), 
             xytext=(f1 + 0.1, A1 + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

plt.annotate(f'A2: {A2} @ {f2:.3f} Hz', 
             xy=(f2, A2), 
             xytext=(f2 + 0.1, A2 + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10)

plt.tight_layout()
plt.show()

# --- Kết quả in ra Console ---
print("-" * 50)
print(f"Tổng số mẫu N: {N}")
print(f"Thời gian tính toán FFT (O(N log N)): {calculation_time:.3f} ms")
print(f"Biên độ tại f1={f1:.3f} Hz: {magnitude[int(f1/df)]:.2f} (Kỳ vọng: {A1})")
print(f"Biên độ tại f2={f2:.3f} Hz: {magnitude[int(f2/df)]:.2f} (Kỳ vọng: {A2})")
print("\nSo sánh với DFT (N=200): FFT (N=256) sẽ NHANH HƠN đáng kể.")