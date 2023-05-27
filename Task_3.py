from Task_2 import *

def encode_D_frame(image_array, I_frame, quantization_matrix, num_blocks_height, num_blocks_width, block_size=8, decimals=0):
     # Split the image into blocks
    image_blocks = split_image_into_blocks(image_array, num_blocks_height, num_blocks_width, block_size)

    # Transform coding
    dct_coefficients_blocks = transform_coding(image_blocks, decimals=decimals)

    # Quantization
    quantized_blocks = quantization(dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Inverse quantization
    inverse_dct_coefficients_blocks = inverse_quantization(quantized_blocks, quantization_matrix, decimals=decimals)

    # Inverse transform coding
    inverse_image_blocks = inverse_transform_coding(inverse_dct_coefficients_blocks, decimals=decimals)

    # Split I-frame into blocks
    I_frame_blocks = split_image_into_blocks(I_frame, num_blocks_height, num_blocks_width, block_size)

    # Compute the difference between the D-frame and the I-frame
    difference_blocks =  I_frame_blocks - inverse_image_blocks

    # Transform coding for the difference
    D_frame_dct_coefficients_blocks = transform_coding(difference_blocks, decimals=decimals)

    # Quantization for the difference
    D_frame_quantized_blocks = quantization(D_frame_dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Zigzag scan
    zigzad_blocks = zigzag_scan(D_frame_quantized_blocks)

    # Entropy coding
    bitstream, codec = entropy_coding(zigzad_blocks)

    # Return the quantized blocks
    return bitstream, codec


def decode_D_frame(bitstream, I_frame, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size=8, decimals=0):
    # Entropy decoding
    zigzad_blocks = inverse_entropy_coding(bitstream, codec)

    # Inverse zigzag scan
    D_frame_quantized_blocks = inverse_zigzag_scan(zigzad_blocks, num_blocks_height, num_blocks_width, block_size)

    # Inverse quantization for the difference
    D_frame_dct_coefficients_blocks = inverse_quantization(D_frame_quantized_blocks, quantization_matrix, decimals=decimals)

    # Inverse transform coding for the difference
    difference_blocks = inverse_transform_coding(D_frame_dct_coefficients_blocks, decimals=decimals)

    # Split I-frame into blocks
    I_frame_blocks = split_image_into_blocks(I_frame, num_blocks_height, num_blocks_width, block_size)

    # Compute the D-frame
    D_frame_blocks = I_frame_blocks - difference_blocks

    # Reconstruct the image
    D_frame = merge_blocks_into_image(D_frame_blocks)

    # Return the D-frame
    return D_frame


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

    compressed_frames = []
    t = time.time()
    for i, frame in enumerate(frames):
        print(f"Encoding frame {i}")
        if i % 5 == 0:
            bitstream, codec = encode(frame, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
            I_frame = decode(bitstream, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
        else:
            bitstream, codec = encode_D_frame(frame, I_frame, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
        
        compressed_frames.append((bitstream, codec)) 

    decompressed_frames = []
    for i, compressed_frame in enumerate(compressed_frames):
        print(f"Decoding frame {i}")
        bitstream, codec = compressed_frame

        if i % 5 == 0:     
            I_frame = decode(bitstream, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals) 
            decompressed_frames.append(I_frame)
        else:
            decoded_frame = decode_D_frame(bitstream, I_frame, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
            decompressed_frames.append(decoded_frame)
    
    print(f"Time: {round(time.time() - t, 2)}")
    create_y4m_video("media/output/foreman_qcif_mono_task3.y4m", decompressed_frames, metadata)