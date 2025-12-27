import numpy as np
import matplotlib.pyplot as plt

# --- 1. Thiết lập Tham số Lấy mẫu ---
Fs = 10000.0 # Tần số lấy mẫu (Hz)
T = 1.0      # Thời gian tín hiệu (giây)
N = int(Fs * T) # Tổng số mẫu
t = np.linspace(0.0, T, N, endpoint=False)
df = Fs / N

# --- 2. Tạo Tín hiệu Hỗn hợp (Giọng nói + Nhiễu) ---

# A. Giọng nói (Mô phỏng 3 tần số cơ bản của âm thanh)
F_speech = [300.0, 500.0, 700.0]
signal_speech = np.zeros_like(t)
for f in F_speech:
    signal_speech += 1.0 * np.sin(2.0 * np.pi * f * t) # Biên độ lớn (1.0)

# B. Nhiễu Tạp âm Trắng (Mô phỏng tạp âm đường phố ngẫu nhiên)
# Sử dụng nhiễu Gaussian với biên độ nhỏ
noise_white = 0.5 * np.random.randn(N) 

# C. Nhiễu Cố định (Mô phỏng tiếng động cơ/tiếng ồn băng thông hẹp)
F_engine_noise = 100.0
noise_engine = 2.0 * np.sin(2.0 * np.pi * F_engine_noise * t) # Biên độ cực lớn (2.0)

# Tín hiệu BỊ NHIỄM (Đầu vào)
signal_input = signal_speech + noise_white + noise_engine

# --- 3. PHÂN TÍCH FFT ---
fft_output = np.fft.fft(signal_input)
fft_magnitude = np.abs(fft_output)

# --- 4. LỌC TÍN HIỆU bằng Ngưỡng Biên độ (Thresholding) ---

# Đặt Ngưỡng Lọc: 
# Tín hiệu giọng nói và nhiễu cố định có biên độ ~ 1.0-2.0.
# Nhiễu tạp âm trắng có biên độ nhỏ hơn.
# Chúng ta sẽ giữ lại 3 đỉnh giọng nói và 1 đỉnh nhiễu cố định. 
# Giả sử ta biết tín hiệu mong muốn nằm dưới 1000 Hz.
F_max_analyze = 1000 
idx_max = int(F_max_analyze / df)

# Tìm Ngưỡng (Threshold): Chọn ngưỡng bằng 5 lần biên độ nhiễu tạp âm trắng trung bình.
# Biên độ trung bình của White Noise là rất nhỏ, ngưỡng 50 là an toàn.
THRESHOLD = 50.0 

fft_filtered = np.copy(fft_output)

# 1. Lọc nhiễu ngẫu nhiên và nhiễu cố định > 1000 Hz
# Xóa tất cả các thành phần tần số cao > 1000 Hz (Giả định Giọng nói nằm trong 1000 Hz)
fft_filtered[idx_max : N // 2] = 0.0
fft_filtered[N - idx_max :] = 0.0

# 2. Áp dụng Thresholding trong dải 0-1000 Hz
for i in range(1, idx_max):
    # Nếu biên độ của bin tần số (i) nhỏ hơn Ngưỡng, ta xóa nó (loại bỏ White Noise)
    if np.abs(fft_filtered[i]) < THRESHOLD:
        fft_filtered[i] = 0.0
        # Xóa cả bin đối xứng
        fft_filtered[N - i] = 0.0


# --- 5. TÁI TẠO IFFT ---
signal_output = np.fft.ifft(fft_filtered)
signal_output = np.real(signal_output)


# --- 6. Trực quan hóa Kết quả (Tối ưu hóa Hiển thị) ---
# CHỈ VẼ 0.1 GIÂY để thấy rõ sự khác biệt
VIEW_LIMIT = int(Fs * 0.1) 

plt.figure(figsize=(15, 12))

# 6a. BIỂU ĐỒ 1: Tín hiệu Gốc (Giọng nói bị chồng tạp âm lớn)
plt.subplot(3, 1, 1)
plt.plot(t[:VIEW_LIMIT], signal_input[:VIEW_LIMIT], color='red') 
plt.title('1. Tín hiệu Bị Nhiễm: Giọng nói + Tạp âm Trắng + Nhiễu cố định 100 Hz')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True, linestyle='--')

# 6b. BIỂU ĐỒ 2: Phổ Tần số (Trước và Sau Lọc)
freq = np.fft.fftfreq(N, 1/Fs)[:N // 2]
plt.subplot(3, 1, 2)
# Phổ Gốc (chỉ lấy biên độ lớn)
plt.plot(freq, fft_magnitude[:N // 2], 'o', label='Phổ Gốc', markersize=2, color='gray', alpha=0.5) 
# Phổ Đã Lọc (chỉ còn các đỉnh lớn)
plt.plot(freq, np.abs(fft_filtered[:N // 2]), 'o', label='Phổ Đã Lọc', markersize=4, color='orange')

plt.axvline(F_engine_noise, color='purple', linestyle='--', label=f'Nhiễu Cố định (100 Hz)')
plt.title(f'2. Phổ Tần số: Lọc bằng Ngưỡng Biên độ (Threshold = {THRESHOLD:.1f})')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.xlim(0, 1500) 
plt.legend()
plt.grid(True, linestyle='--')

# 6c. BIỂU ĐỒ 3: Tín hiệu Đã Lọc (Chỉ còn Giọng nói)
plt.subplot(3, 1, 3)
plt.plot(t[:VIEW_LIMIT], signal_output[:VIEW_LIMIT], color='blue')
plt.title('3. Tín hiệu Đã Lọc: Chỉ còn tín hiệu Giọng nói')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show()