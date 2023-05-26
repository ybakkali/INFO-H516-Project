from Task_1 import *

video_path = "task_2/foreman_qcif_mono.y4m"
#video_path = "task_2/foreman_qcif.y4m"
#video_path = "task_2/foreman_cif.y4m"


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

if __name__ == "__main__":

    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    frames, metadata = read_y4m_video("task_2/foreman_qcif_mono.y4m")
    #frames = np.stack(frames, axis=0)
    compressed_frames = []

    for i, frame in enumerate(frames):
        print(f'Frame {i}')
        encode_frame, codec, shape = encode(frame, quantization_matrix)
        decode_frame = decode(encode_frame, codec, shape, quantization_matrix)
        compressed_frames.append(decode_frame)

    create_y4m_video("task_2/foreman_qcif_mono_compressed.y4m", compressed_frames, metadata)