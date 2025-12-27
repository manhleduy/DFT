import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến tệp âm thanh đã cung cấp
AUDIO_FILE_PATH = "FFT-test-2.mp3" # <-----------

# --- 1. Tải và Lấy Mẫu Tín hiệu Âm thanh ---

try:
    # Tải tệp âm thanh. y là mảng biên độ, Fs là tần số lấy mẫu.
    # Sử dụng mono=True để chuyển về kênh đơn nếu cần.
    y, Fs = librosa.load(AUDIO_FILE_PATH, sr=None, mono=True)
    N = len(y) # Tổng số mẫu
    
    print("-" * 50)
    print(f"Tải tệp thành công: {AUDIO_FILE_PATH}")
    print(f"Tần số lấy mẫu (Fs): {Fs} Hz")
    print(f"Tổng số mẫu (N): {N}")
    print(f"Thời lượng: {N/Fs:.2f} giây")
    print("-" * 50)

except Exception as e:
    print(f"LỖI: Không thể tải tệp âm thanh. Vui lòng kiểm tra đường dẫn và định dạng tệp.")
    print(f"Chi tiết lỗi: {e}")
    exit()

# --- 2. Phân tích FFT ---
# Thực hiện FFT trên toàn bộ tín hiệu
fft_output = np.fft.fft(y)

# --- 3. Chuẩn hóa và Ánh xạ Tần số ---

# Chỉ lấy nửa đầu của phổ (phổ dương)
magnitude = np.abs(fft_output[:N // 2]) 

# Chuẩn hóa biên độ (thường được bỏ qua trong phân tích âm thanh, 
# nhưng ta giữ lại để biểu thị cường độ tương đối)
magnitude = magnitude / N

# Tạo mảng tần số thực
freq = np.fft.fftfreq(N, 1/Fs)[:N // 2]


# --- 4. Trực quan hóa Kết quả ---

plt.figure(figsize=(15, 8))

# Biểu đồ 1: Sóng Âm thanh (Miền Thời gian)
plt.subplot(2, 1, 1)
# Chỉ hiển thị một đoạn nhỏ (ví dụ: 0.2s đầu tiên) để thấy chi tiết
time_samples = np.linspace(0, N/Fs, N)
plt.plot(time_samples, y, color='blue')
plt.title('1. Tín hiệu Âm thanh (Miền Thời gian)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
# Giới hạn hiển thị trục x để dễ quan sát sóng (chỉ 0.5 giây)
plt.xlim(0, 0.5) 
plt.grid(True, linestyle='--')

# Biểu đồ 2: Phổ Tần số (Miền Tần số)
plt.subplot(2, 1, 2)
# Chuyển đổi biên độ sang dB (Decibel) để dễ quan sát dải động lớn của âm thanh
# (Sử dụng 20*log10(magnitude) cho biên độ)
magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max) 

plt.plot(freq, magnitude_db, color='red')
plt.title(f'2. Phổ Tần số của Tệp Âm thanh (Sử dụng FFT) - Fs={Fs} Hz')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Cường độ (dB)')
# Giới hạn hiển thị phổ (thường đến khoảng 5000 Hz cho giọng nói/nhạc cụ)
plt.xlim(0, 5000) 
plt.ylim(np.max(magnitude_db) - 60, np.max(magnitude_db)) # Chỉ hiển thị dải động 60 dB
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# --- Kết luận Phân tích ---
# Tìm tần số có cường độ lớn nhất (trừ tần số DC = 0 Hz)
peak_index = np.argmax(magnitude[1:]) + 1
peak_freq = freq[peak_index]
peak_db = magnitude_db[peak_index]

print("-" * 50)
print(f"Phân tích nhanh: Tần số có cường độ mạnh nhất là: {peak_freq:.2f} Hz ({peak_db:.2f} dB)")
print("=> Biểu đồ 2 cho thấy cấu trúc phổ của tín hiệu. Các đỉnh chính là các tần số chủ đạo trong âm thanh.")