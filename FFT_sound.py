import numpy as np
import matplotlib.pyplot as plt

# --- 1. Thiết lập Tham số Âm thanh ---
Fs = 44100.0  # Tần số lấy mẫu tiêu chuẩn (Hz)
T = 2.0       # Thời gian tín hiệu (giây)
N = int(Fs * T) # Tổng số mẫu
t = np.linspace(0.0, T, N, endpoint=False)

# --- 2. Định nghĩa Tần số Cơ bản (C4-E4-G4) ---
F_C4 = 261.63 
F_E4 = 329.63
F_G4 = 392.00

# Điều chỉnh tần số để tránh Rò rỉ Phổ
df = Fs / N
F_C4_adj = round(F_C4 / df) * df
F_E4_adj = round(F_E4 / df) * df
F_G4_adj = round(F_G4 / df) * df


# --- 3. Tạo Tín hiệu Hợp âm Phức tạp (Bao gồm Họa âm) ---
# Nốt C4 (Tần số cơ bản + Họa âm)
signal_C4 = 1.0 * np.sin(2.0 * np.pi * F_C4_adj * t) + \
            0.5 * np.sin(2.0 * np.pi * (2 * F_C4_adj) * t)

# Nốt E4 (Tần số cơ bản)
signal_E4 = 0.8 * np.sin(2.0 * np.pi * F_E4_adj * t)

# Nốt G4 (Tần số cơ bản + Họa âm)
signal_G4 = 0.7 * np.sin(2.0 * np.pi * F_G4_adj * t) + \
            0.3 * np.sin(2.0 * np.pi * (3 * F_G4_adj) * t)

# Tín hiệu Hợp âm Tổng hợp
chord_signal = signal_C4 + signal_E4 + signal_G4


# --- 4. Thực hiện FFT (Ứng dụng) ---
fft_output = np.fft.fft(chord_signal)

# --- 5. Chuẩn hóa và Ánh xạ Tần số ---
magnitude = np.abs(fft_output)
magnitude = magnitude[:N // 2]  

magnitude[1:] = 2 * magnitude[1:]
magnitude = magnitude / N 

freq = np.fft.fftfreq(N, 1/Fs)[:N // 2]


# --- 6. Trực quan hóa Kết quả ---

plt.figure(figsize=(14, 8))

# Biểu đồ 1: Sóng Hợp âm (Miền Thời gian)
plt.subplot(2, 1, 1)
# Chỉ vẽ một đoạn nhỏ (20ms) để dễ quan sát sóng phức tạp
plt.plot(t[:int(Fs/50)], chord_signal[:int(Fs/50)], color='blue') 
plt.title('1. Tín hiệu Hợp âm C Major (Đoạn 20ms Miền Thời gian)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True, linestyle='--')

# Biểu đồ 2: Phổ Tần số (Phân tích Hợp âm bằng FFT)
plt.subplot(2, 1, 2)
plt.plot(freq, magnitude, color='red') 
plt.title('2. Phổ Tần số của Hợp âm C Major (Phân tích FFT)')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ (Magnitude)')
plt.xlim(0, 1000) # Giới hạn dải quan sát đến 1000 Hz
plt.xticks(np.arange(0, 1001, 100))
plt.grid(True, linestyle='--')

# Sửa lỗi: Danh sách annotations chỉ có (tần số, nhãn)
annotations = [
    (F_C4_adj, "C4 (261.6 Hz)"), 
    (F_E4_adj, "E4 (329.6 Hz)"), 
    (F_G4_adj, "G4 (392.0 Hz)"), 
    (2 * F_C4_adj, "Họa âm C4 (523.3 Hz)"),
    # Loại bỏ giá trị thứ ba gây lỗi (0.3)
    (3 * F_G4_adj, "Họa âm G4 (1176.0 Hz)") 
]

for f, label in annotations:
    if f < 1000: # Chỉ chú thích trong phạm vi hiển thị (0-1000 Hz)
        amp = magnitude[int(f/df)]
        if amp > 0.05: # Chỉ chú thích các đỉnh đáng kể
            plt.annotate(label, 
                         xy=(f, amp), 
                         xytext=(f + 50, amp * 1.5),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                         fontsize=9)

plt.tight_layout()
plt.show()

# --- Kết quả in ra Console ---
print("-" * 50)
print("Phân tích Phổ Hợp âm C Major (Sử dụng FFT):")
print(f"Độ phân giải tần số (df): {df:.3f} Hz")
print(f"C4: Tần số thực tế: {F_C4_adj:.2f} Hz, Biên độ FFT: {magnitude[int(F_C4_adj/df)]:.2f} (Kỳ vọng: 1.0)")
print(f"E4: Tần số thực tế: {F_E4_adj:.2f} Hz, Biên độ FFT: {magnitude[int(F_E4_adj/df)]:.2f} (Kỳ vọng: 0.8)")
print(f"G4: Tần số thực tế: {F_G4_adj:.2f} Hz, Biên độ FFT: {magnitude[int(F_G4_adj/df)]:.2f} (Kỳ vọng: 0.7)")
print("=> Chương trình đã được sửa lỗi và sẵn sàng hoạt động.")