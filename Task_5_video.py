from moviepy.editor import VideoFileClip
import os
from Task_1 import *
from Task_2 import *
from Task_3 import *

def rate_distortion_curve_task5(input_path, output_path, codec):
      
    psnr_values = []
    bps_values = []

    with VideoFileClip(input_path, audio=False, fps_source='fps') as video_clip:

        frames = [frame[:, :, 0] for frame in video_clip.iter_frames()][30]

        size =  os.path.getsize(input_path)
        duration = video_clip.duration
        bitrate = (size * 8 / 1000) / duration

        # numpy array of 2 powers
        # ratios = np.array([2 ** i for i in range(1, 6)])
        # Calculate the target bitrates based on the compression ratios
        # target_bitrates = bitrate / ratios

        crf_levels = np.arange(0, 52, 5)

        for crf in crf_levels:

            print(f"Constant Rate Factor: {crf}")

            # Set the target bitrate (in kbps)
            video_clip.write_videofile(output_path, codec=codec, logger=None, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(crf)])

            with VideoFileClip(output_path, audio=False, fps_source='fps') as video_clip_compressed:

                compressed_frames = [frame[:, :, 0] for frame in video_clip_compressed.iter_frames()][30]

                psnr_sum = sum([PSNR(frame, compressed_frame) for frame, compressed_frame in zip(frames, compressed_frames)])

                # Calculate average PSNR and BPS
                psnr = psnr_sum / len(frames)
                bps = os.path.getsize(output_path) * 8 / duration

            # Store PSNR and BPS values
            psnr_values.append(round(psnr, 2))
            bps_values.append(round(bps/ 1000, 2))

    return psnr_values, bps_values, crf_levels


def plot_curves():

    psnr_values, bps_values, quantization_levels = rate_distortion_curve_task2(frames[:30], quantization_matrix, 30, num_blocks_height, num_blocks_width, block_size, decimals)
    psnr_values_D_frame, bps_values_D_frame, quantization_levels_D_frame = rate_distortion_curve_task3(frames[:30], 5, quantization_matrix, 30, num_blocks_height, num_blocks_width, block_size, decimals)
    psnr_values_h264, bps_values_h264, ratios_h264 = rate_distortion_curve_task5(input_video_path, output_video_path_h264, 'libx264')
    psnr_values_h265, bps_values_h265, ratios_h265 = rate_distortion_curve_task5(input_video_path, output_video_path_h265, 'libx265')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bps_values, y=psnr_values, mode='lines+markers', name='Task 2', text=quantization_levels))
    fig.add_trace(go.Scatter(x=bps_values_D_frame, y=psnr_values_D_frame, mode='lines+markers', name='Task 3', text=quantization_levels_D_frame))
    fig.add_trace(go.Scatter(x=bps_values_h264, y=psnr_values_h264, mode='lines+markers', name='H.264', text=ratios_h264))
    fig.add_trace(go.Scatter(x=bps_values_h265, y=psnr_values_h265, mode='lines+markers', name='H.265', text=ratios_h265))
    fig.update_layout(title='Rate-Distortion Curve', xaxis_title='BitsPerSecond (kbps)', yaxis_title='PSNR (dB)')
    fig.show()


if __name__ == "__main__":
        
    input_video_path = 'media/input/foreman_qcif_mono.y4m'
    output_video_path_h264 = 'media/output/foreman_qcif_mono_h264.mp4'
    output_video_path_h265 = 'media/output/foreman_qcif_mono_h265.mp4'

    frames, metadata = read_y4m_video(input_video_path)
    frame_height, frame_width, block_size = int(metadata["H"]), int(metadata["W"]), 8
    num_blocks_height = frame_height // block_size
    num_blocks_width = frame_width // block_size
    decimals = 0
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    plot_curves()


