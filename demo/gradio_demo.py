import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import gradio as gr
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from architecture import AEC

def aec_inference_interface(ref_audio_path, mic_audio_path):
    if ref_audio_path is None or mic_audio_path is None:
        return None, None, "Vui lòng cung cấp đủ cả 2 file âm thanh."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SR = 16000
    
    # Load Model (Đảm bảo file .pth nằm cùng thư mục)
    model = AEC(d_model=128, n_fft=512).to(device)
    model_path = "aec_v2_step_2900.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load Audio
    y_ref, _ = librosa.load(ref_audio_path, sr=SR)
    y_mic, _ = librosa.load(mic_audio_path, sr=SR)
    
    # Căn chỉnh độ dài
    min_len = min(len(y_mic), len(y_ref))
    y_mic, y_ref = y_mic[:min_len], y_ref[:min_len]
    
    # Inference logic
    n_fft, hop_length, win_length = 512, 160, 320
    window = torch.hann_window(win_length).to(device)
    
    def to_stft(y):
        y_torch = torch.from_numpy(y).float().to(device)
        stft_complex = torch.stft(y_torch, n_fft=n_fft, hop_length=hop_length, 
                                  win_length=win_length, window=window, 
                                  center=True, return_complex=True)
        return torch.view_as_real(stft_complex).unsqueeze(0)

    with torch.no_grad():
        mic_stft = to_stft(y_mic)
        ref_stft = to_stft(y_ref)
        est_stft = model(mic_stft, ref_stft)
        est_complex = torch.complex(est_stft[..., 0], est_stft[..., 1]).squeeze(0)
        y_est = torch.istft(est_complex, n_fft=n_fft, hop_length=hop_length, 
                            win_length=win_length, window=window, center=True)
    
    y_output = y_est.cpu().numpy()
    output_path = "cleaned_audio.wav"
    sf.write(output_path, y_output, SR)

    # Vẽ đồ thị so sánh
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(y_ref, color='blue')
    ax[0].set_title("Reference Signal")
    ax[1].plot(y_mic, color='red')
    ax[1].set_title("Mic Input (Echo + Voice)")
    ax[2].plot(y_output, color='green')
    ax[2].set_title("AEC Output (Cleaned)")
    plt.tight_layout()
    
    return fig, output_path

# ==========================================
# 3. THIẾT LẬP GIAO DIỆN GRADIO
# ==========================================
demo = gr.Interface(
    fn=aec_inference_interface,
    inputs=[
        gr.Audio(label="1. Tải lên file Reference (Âm thanh phát ra loa)", type="filepath"),
        gr.Audio(label="2. Thu âm từ Mic hoặc Tải lên file thu được", type="filepath")
    ],
    outputs=[
        gr.Plot(label="So sánh dạng sóng (Waveforms)"),
        gr.Audio(label="Kết quả sau khi khử Echo (AEC Output)")
    ],
    title="Demo Mô hình AEC Conformer",
    description="Tải lên file nhạc tham chiếu và âm thanh từ microphone để khử tiếng vọng thực tế.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()