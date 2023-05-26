from Task_2 import *

def encode_D_frame(image_array, I_frame, quantization_matrix, block_size=8, decimals=0):
     # Split the image into blocks
    image_blocks = split_image_into_blocks(image_array, block_size)

    # Transform coding
    dct_coefficients_blocks = transform_coding(image_blocks, decimals=decimals)

    # Quantization
    quantized_blocks = quantization(dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Inverse quantization
    inverse_dct_coefficients_blocks = inverse_quantization(quantized_blocks, quantization_matrix, decimals=decimals)

    # Inverse transform coding
    inverse_image_blocks = inverse_transform_coding(inverse_dct_coefficients_blocks, decimals=decimals)

    # Split I-frame into blocks
    I_frame_blocks = split_image_into_blocks(I_frame, block_size)

    # Compute the difference between the D-frame and the I-frame
    difference_blocks =  I_frame_blocks - inverse_image_blocks

    # Transform coding for the difference
    D_frame_dct_coefficients_blocks = transform_coding(difference_blocks, decimals=decimals)

    # Quantization for the difference
    D_frame_quantized_blocks = quantization(D_frame_dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Zigzag scan
    zigzad_blocks = zigzag_scan(D_frame_quantized_blocks)

    # Entropy coding
    encoded_blocks, codec, shape = entropy_coding(zigzad_blocks)

    # Return the quantized blocks
    return encoded_blocks, codec, shape


def decode_D_frame(D_frame_quantized_blocks, I_frame, codec, shape, quantization_matrix, block_size=8, decimals=0):
    # Entropy decoding
    zigzad_blocks = inverse_entropy_coding(D_frame_quantized_blocks, codec, shape)

    # Inverse zigzag scan
    D_frame_quantized_blocks = inverse_zigzag_scan(zigzad_blocks)

    # Inverse quantization for the difference
    D_frame_dct_coefficients_blocks = inverse_quantization(D_frame_quantized_blocks, quantization_matrix, decimals=decimals)

    # Inverse transform coding for the difference
    difference_blocks = inverse_transform_coding(D_frame_dct_coefficients_blocks, decimals=decimals)

    # Split I-frame into blocks
    I_frame_blocks = split_image_into_blocks(I_frame, block_size)

    # Compute the D-frame
    D_frame_blocks = I_frame_blocks - difference_blocks

    # Reconstruct the image
    D_frame = merge_blocks_into_image(D_frame_blocks)

    # Return the D-frame
    return D_frame



quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

frames, metadata = read_y4m_video("task_2/foreman_qcif_mono.y4m")

compressed_frames = []

for i, frame in enumerate(frames):
    print(f'Frame {i}')
    if i % 5 == 0:
        encode_frame, codec, shape = encode(frame, quantization_matrix)
        I_frame = decode(encode_frame, codec, shape, quantization_matrix)
    else:
        encode_frame, codec, shape = encode_D_frame(frame, I_frame, quantization_matrix)
    
    compressed_frames.append((encode_frame, codec, shape)) 

decompressed_frames = []
for i, frame in enumerate(compressed_frames):
    print(f'Frame {i}')
    compressed_frame, codec, shape = frame

    if i % 5 == 0:     
        I_frame = decode(compressed_frame, codec, shape, quantization_matrix)
        decompressed_frames.append(I_frame)
    else:
        decode_frame = decode_D_frame(compressed_frame, I_frame, codec, shape, quantization_matrix)
        decompressed_frames.append(decode_frame)


create_y4m_video("task_2/foreman_qcif_mono_compressed_task3.y4m", decompressed_frames, metadata)