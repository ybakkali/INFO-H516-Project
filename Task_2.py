from Task_1 import *

def read_y4m_video(video_path):

    frames_list = []
    with open(video_path, "rb") as file:
        # Read the header
        header = file.readline().decode("utf-8").strip().split(" ")
        if header[0] != "YUV4MPEG2":
            raise Exception("Not a y4m file")

        # Read the metadata
        metadata = {}
        for elem in header[1:]: 
            metadata[elem[0]] = elem[1:]


        # Set the frames width and height
        width = int(metadata["W"])
        height = int(metadata["H"])

        frame_header_len = len(b"FRAME\n")
        frame_size = width * height
        
        frame = file.read(frame_header_len + frame_size)
        while frame != b"":
            # Remove the frame header
            frame = frame[frame_header_len:]
            # Convert the frame to a numpy array
            image_array = np.frombuffer(frame, dtype=np.uint8)
            # Reshape the array to a 2D image
            image_array = image_array.reshape((height, width))
            # Add the image to the list of frames
            frames_list.append(image_array)
            # Read the next frame
            frame = file.read(frame_header_len + frame_size)

    return frames_list, metadata


def create_y4m_video(video_path, frames, metadata):
    with open(video_path, "wb") as file:
        # Write the Y4M header
        header = "YUV4MPEG2 " + " ".join(f"{k}{v}" for k, v in metadata.items()) + "\n"
        file.write(header.encode("utf-8"))

        # Write the frames
        for frame in frames:
            # Flatten the frame
            frame = frame.flatten()
            # Convert the frame to bytes
            frame_bytes = frame.tobytes()
            # Write the frame header
            file.write(b"FRAME\n")
            # Write the frame data
            file.write(frame_bytes)

def rate_distortion_curve_task2(frames, quantization_matrix, fps, num_blocks_height, num_blocks_width, block_size, decimals):
      
    psnr_values = []
    bps_values = []
    quantization_levels = [1] + [2 ** i for i in range(1, 6)]
    for quantization_level in quantization_levels:

        print(f"Quantization level: {quantization_level}")
        psnr_sum, size_sum = 0, 0

        for frame in frames:

            # Encode frame
            bitstream, codec = encode(frame, quantization_matrix * quantization_level, num_blocks_height, num_blocks_width, block_size, decimals)

            # Calculate frame size in bits
            size_sum += len(bitstream) * 8

            # Decode frame
            decoded_frame = decode(bitstream, codec, quantization_matrix * quantization_level, num_blocks_height, num_blocks_width, block_size, decimals)

            # Calculate PSNR
            psnr_sum += PSNR(frame, decoded_frame)

        # Calculate average PSNR and BPS
        psnr = psnr_sum / len(frames)
        bps = size_sum / (len(frames) / fps)

        # Store PSNR and BPS values
        psnr_values.append(round(psnr, 2))
        bps_values.append(round(bps/ 1000, 2))

    return psnr_values, bps_values, quantization_levels

def plot_rate_distortion_curve(psnr_values, bps_values, levels):
    fig = go.Figure(data=go.Scatter(x=bps_values, y=psnr_values, mode='lines+markers', text=levels))
    fig.update_layout(title='Rate-Distortion Curve', xaxis_title='BitsPerSecond (kbps)', yaxis_title='PSNR (dB)')
    fig.show()

if __name__ == "__main__":

    filename = "media/input/foreman_qcif_mono.y4m"
    frames, metadata = read_y4m_video(filename)
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

    # encoded_frames = []
    # t = time.time()
    # for i, frame in enumerate(frames):
    #     print(f"Encoding frame {i}")
    #     bitstream, codec = encode(frame, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
    #     encoded_frames.append((bitstream, codec))

    # compressed_frames = []
    # for i, encoded_frame in enumerate(encoded_frames):
    #     print(f"Decoding frame {i}")
    #     bitstream, codec = encoded_frame
    #     decoded_frame = decode(bitstream, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
    #     compressed_frames.append(decoded_frame)
    
    # print(f"Time: {round(time.time() - t, 2)}")
    # create_y4m_video("media/output/foreman_qcif_mono_task2.y4m", compressed_frames, metadata)

    psnr_values, bps_values, quantization_levels = rate_distortion_curve_task2(frames[:30], quantization_matrix, 30, num_blocks_height, num_blocks_width, block_size, decimals)
    plot_rate_distortion_curve(psnr_values, bps_values, quantization_levels)