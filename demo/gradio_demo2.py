import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import gradio as gr
import sounddevice as sd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from architecture import AEC


def run_aec_full_flow(ref_audio_path):
    if ref_audio_path is None:
        return None, None, None, "Vui lòng upload file Reference."

    SR = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đọc file Reference
    y_ref, _ = librosa.load(ref_audio_path, sr=SR)
    
    # 2. Play và Record đồng thời
    print("--- Đang phát Ref và thu Mic... ---")
    recorded_mic = sd.playrec(y_ref, samplerate=SR, channels=1)
    sd.wait()
    y_mic = recorded_mic.flatten()

    # Lưu lại file Mic vừa thu để hiển thị ra Gradio
    mic_recorded_path = "mic_captured_raw.wav"
    sf.write(mic_recorded_path, y_mic, SR)

    # 3. Load Model & Inference
    model = AEC(d_model=128, n_fft=512).to(device)
    model_path = "aec_v2_step_2900.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_fft, hop_length, win_length = 512, 160, 320
    window = torch.hann_window(win_length).to(device)
    
    def to_stft(y):
        y_t = torch.from_numpy(y).float().to(device)
        stft = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, 
                          win_length=win_length, window=window, 
                          center=True, return_complex=True)
        return torch.view_as_real(stft).unsqueeze(0)

    with torch.no_grad():
        min_len = min(len(y_mic), len(y_ref))
        y_mic_f, y_ref_f = y_mic[:min_len], y_ref[:min_len]
        
        mic_s = to_stft(y_mic_f)
        ref_s = to_stft(y_ref_f)
        est_s = model(mic_s, ref_s)
        
        est_c = torch.complex(est_s[..., 0], est_s[..., 1]).squeeze(0)
        y_est = torch.istft(est_c, n_fft=n_fft, hop_length=hop_length, 
                            win_length=win_length, window=window, center=True)
    
    y_out = y_est.cpu().numpy()
    out_path = "aec_cleaned_output.wav"
    sf.write(out_path, y_out, SR)

    # 4. Vẽ đồ thị so sánh
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(y_ref_f, color='blue', alpha=0.6); axes[0].set_title("1. Reference (Phát ra loa)")
    axes[1].plot(y_mic_f, color='red', alpha=0.6); axes[1].set_title("2. Mic Input (Âm thanh thu được - Có tiếng vọng)")
    axes[2].plot(y_out, color='green'); axes[2].set_title("3. AEC Output (Âm thanh đã lọc sạch)")
    plt.tight_layout()

    return mic_recorded_path, out_path, fig

# ==========================================
# 3. GIAO DIỆN GRADIO
# ==========================================
with gr.Blocks(title="AEC Conformer Demo") as demo:
    gr.Markdown("## Mô phỏng AEC: Tự động Thu/Phát & Khử Echo")
    
    with gr.Row():
        with gr.Column():
            ref_in = gr.Audio(label="Tải lên file Reference (Âm thanh nền)", type="filepath")
            run_btn = gr.Button("BẮT ĐẦU: PHÁT NHẠC & THU MIC", variant="primary")
        
    gr.Markdown("### Kết quả thu được và xử lý:")
    with gr.Row():
        mic_audio_out = gr.Audio(label="Mic đã thu (Có Echo)", type="filepath")
        cleaned_audio_out = gr.Audio(label="Kết quả sau AEC (Sạch)", type="filepath")
        
    plot_out = gr.Plot(label="So sánh Waveforms")

    # Link button click
    run_btn.click(
        fn=run_aec_full_flow,
        inputs=ref_in,
        outputs=[mic_audio_out, cleaned_audio_out, plot_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)