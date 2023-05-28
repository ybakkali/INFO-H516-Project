import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import groupby
import time
from dahuffman import HuffmanCodec


def open_raw_image(file_path, image_width, image_height):

    # Read the raw file as binary
    with open(file_path, 'rb') as f:
        # Read the binary data
        raw_data = f.read()
    
    # Convert the binary data to an array
    image_array = np.frombuffer(raw_data, dtype=np.uint8)
    
    # Reshape the array to a 2D image
    image_array = image_array.reshape((image_height, image_width))
    
    return image_array

def split_image_into_blocks(image, num_blocks_height, num_blocks_width, block_size):

    # Create an empty array to store the blocks
    blocks = np.zeros((num_blocks_height, num_blocks_width, block_size, block_size), dtype=np.uint8)

    # Split the image into blocks
    for i, j in np.ndindex(num_blocks_height, num_blocks_width):
        blocks[i, j] = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
    
    return blocks

def merge_blocks_into_image(image_blocks):
    
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_size, _ = image_blocks.shape

    # Calculate the image dimensions
    image_height = num_blocks_height * block_size
    image_width = num_blocks_width * block_size

    # Create an empty array to store the reconstructed image
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Merge the blocks into the image
    for i, j in np.ndindex(num_blocks_height, num_blocks_width):
        image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.clip(image_blocks[i, j], 0, 255)
    
    return image

def dct_1D(array, normalize=True):
    
    N = len(array)
    dct_coef = np.zeros(N)

    # Iterate over the 1D array
    for k in range(N):
            
            if normalize:
                # Normalization factors
                scale_factor = np.sqrt(1/(4*N)) if k == 0 else np.sqrt(1/(2*N))
            else:
                scale_factor = 1
            
            # Compute the DCT coefficient
            dct_coef[k] = scale_factor * 2 * np.sum(array * np.cos(((2 * np.arange(N) + 1) * k * np.pi) / (2*N)))

    return dct_coef

def dct_2D(block):
    
    M, N = block.shape
    # Create an empty array to store the DCT coefficients
    dct_block = np.zeros((M, N))

    # Compute the 1D DCT for each row
    for i in range(M):
        dct_block[i,:] = dct_1D(block[i,:])
    
    # Compute the 1D DCT for each column
    for j in range(N):
        dct_block[:,j] = dct_1D(dct_block[:,j])

    return dct_block

def idct_1D(array, normalize=True):
        
    N = len(array)
    dct_coef = np.zeros(N)

    for k in range(N):
            
            if normalize:
                # Normalization factors
                scale_factor = 1 / np.sqrt(N)
                scale_factor_2 = (np.sqrt(2)/ 2) * scale_factor
            else:
                scale_factor = 1
                scale_factor_2 = 1
            
            # Compute the DCT coefficient
            dct_coef[k] = scale_factor * array[0] + scale_factor_2 * 2 * np.sum(array[1:] * np.cos(((2*k + 1) * np.arange(1, N) * np.pi) / (2*N)))

    return dct_coef

def idct_2D(block):
    
    M, N = block.shape

    # Create an empty array to store the DCT coefficients
    dct_block = np.zeros((M, N))

    # Compute the 1D DCT for each row
    for i in range(M):
        dct_block[i,:] = idct_1D(block[i,:])
    
    # Compute the 1D DCT for each column
    for j in range(N):
        dct_block[:,j] = idct_1D(dct_block[:,j])

    return dct_block

def transform_coding(image_blocks, decimals=0):

    # Create an empty array to store the DCT coefficients blocks
    dct_coefficients_blocks = np.zeros(image_blocks.shape)

    # Iterate over the blocks
    for i, j in np.ndindex(image_blocks.shape[:2]):
        # Calculate the DCT coefficients
        dct_coefficients_blocks[i, j] = dct_2D(image_blocks[i, j])

    return np.round(dct_coefficients_blocks, decimals=decimals)

def inverse_transform_coding(dct_coefficients_blocks, decimals=0):

    # Create an empty array to store the DCT coefficients blocks
    image_blocks = np.zeros(dct_coefficients_blocks.shape)

    # Iterate over the blocks
    for i, j in np.ndindex(dct_coefficients_blocks.shape[:2]):
        # Calculate the DCT coefficients
        image_blocks[i, j] = idct_2D(dct_coefficients_blocks[i, j])

    return np.round(image_blocks, decimals=decimals)

def quantization(dct_coefficients_blocks, quantization_matrix, decimals=0):

    # Create an empty array to store the quantized blocks
    quantized_blocks = np.zeros(dct_coefficients_blocks.shape)

    # Iterate over the blocks
    for i, j in np.ndindex(dct_coefficients_blocks.shape[:2]):
        # Calculate the quantized coefficients
        quantized_blocks[i, j] = dct_coefficients_blocks[i, j] / quantization_matrix

    return np.round(quantized_blocks, decimals=decimals)

def inverse_quantization(quantized_blocks, quantization_matrix, decimals=0):

    # Create an empty array to store the quantized blocks
    dct_coefficients_blocks = np.zeros(quantized_blocks.shape)

    # Iterate over the blocks
    for i, j in np.ndindex(quantized_blocks.shape[:2]):
        # Calculate the quantized coefficients
        dct_coefficients_blocks[i, j] = quantized_blocks[i, j] * quantization_matrix

    return np.round(dct_coefficients_blocks, decimals=decimals)

def zigzag_scan_block_to_array(block):
    block_size = block.shape[0]
    row, col = 0, 0
    zigzag_block = []
    direction = 1  # 1 for upward movement, -1 for downward movement

    # Iterate over the zigzag pattern until all elements are traversed
    for i in range(block_size * block_size):

        zigzag_block += [block[row, col]]
        
        # Move upward
        if direction == 1:
            # At the last column of the block
            if col == block_size - 1:
                row, direction = row + 1, -1
            # At the first row of the block
            elif row == 0:
                col, direction = col + 1, -1
            else:
                row, col = row - 1, col + 1

        # Move downward
        else:
            # At the last row of the block
            if row == block_size - 1:
                col, direction = col + 1, 1
            # At the first column of the block
            elif col == 0:
                row, direction = row + 1, 1
            else:
                row, col = row + 1, col - 1

    return zigzag_block

def zigzag_scan_array_to_block(array, block_size):

    block = np.zeros((block_size, block_size))

    row, col = 0, 0
    direction = 1  # 1 for upward movement, -1 for downward movement

    # Iterate over the zigzag pattern to reconstruct the block
    for value in array:
        block[row, col] = value

        # Move upward
        if direction == 1:
            # At the last column of the block
            if col == block_size - 1:
                row, direction = row + 1, -1
            # At the first row of the block
            elif row == 0:
                col, direction = col + 1, -1
            else:
                row, col = row - 1, col + 1

        # Move downward
        else:
            # At the last row of the block
            if row == block_size - 1:
                col, direction = col + 1, 1
            # At the first column of the block
            elif col == 0:
                row, direction = row + 1, 1
            else:
                row, col = row + 1, col - 1

    return block


def zigzag_scan(quanitzed_blocks):

    # Create an empty array to store the zigzag scanned blocks
    zigzag_blocks = []

    # Iterate over the blocks
    for i, j in np.ndindex(quanitzed_blocks.shape[:2]):
        # Parse the block on a zigzag pattern and remove the trailing zeros
        block_1D = np.trim_zeros(zigzag_scan_block_to_array(quanitzed_blocks[i, j]), 'b')
        if not block_1D:
            # Empty block after trimming the zeros
            block_1D = [0]
        zigzag_blocks.extend(block_1D)
        
        # Add infinity as a separator between blocks (EOB)
        zigzag_blocks.append(np.inf)
    
    return zigzag_blocks

def inverse_zigzag_scan(zigzag_blocks, num_blocks_height, num_blocks_width, block_size):

    # Split the bitstream into list of 2D blocks
    separator = np.inf
    inverse_zigzag_blocks = [zigzag_scan_array_to_block(list(group), block_size) for key, group in groupby(zigzag_blocks, lambda x: x != separator) if key]
    # Reshape the list of 2D blocks to a 4D array
    return  np.array(inverse_zigzag_blocks).reshape((num_blocks_height, num_blocks_width, block_size, block_size))

def entropy_coding(zigzag_blocks):

    # Create a Huffman codec object
    codec = HuffmanCodec.from_data(zigzag_blocks)

    # Encode the data
    bitstream = codec.encode(zigzag_blocks)

    return bitstream, codec

def inverse_entropy_coding(bitstream, codec): 
    # Decode the data
    return codec.decode(bitstream)


def encode(image_array, quantization_matrix, num_blocks_height, num_blocks_width, block_size=8, decimals=0):

    # Split the image into blocks
    image_blocks = split_image_into_blocks(image_array, num_blocks_height, num_blocks_width, block_size)

    # Transform coding
    dct_coefficients_blocks = transform_coding(image_blocks, decimals=decimals)

    # Quantization
    quantized_blocks = quantization(dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Zigzag scan
    zigzad_blocks = zigzag_scan(quantized_blocks)

    # Entropy coding
    bitstream, codec = entropy_coding(zigzad_blocks)

    # Return the quantized blocks
    return bitstream, codec

def decode(bitstream, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size=8, decimals=0):

    # Inverse entropy coding
    zigzad_blocks = inverse_entropy_coding(bitstream, codec)

    # Inverse zigzag scan
    quantized_blocks = inverse_zigzag_scan(zigzad_blocks, num_blocks_height, num_blocks_width, block_size)

    # Inverse quantization
    dct_coefficients_blocks = inverse_quantization(quantized_blocks, quantization_matrix, decimals=decimals)

    # Inverse transform coding
    image_blocks = inverse_transform_coding(dct_coefficients_blocks, decimals=decimals)

    # Combine the blocks
    image_array = merge_blocks_into_image(image_blocks)

    # Return the decoded image
    return image_array

def compression_quality(gray_image, encoded_image):
    # Number of bits of the original image
    original_image_size = gray_image.shape[0] * gray_image.shape[1] * 8

    # Number of bits of the compressed image
    compressed_image_size = len(encoded_image) * 8

    # Calculate the compression factor
    compression_factor = original_image_size / compressed_image_size

    # Calculate the bit per pixel (BPP)
    bpp = compressed_image_size / (gray_image.shape[0] * gray_image.shape[1])

    # Print the results
    print('Original image size (bits): ', original_image_size)
    print('Compressed image size (bits): ', compressed_image_size)
    print('Compression factor: ', round(compression_factor,2))
    print('Bit per pixel (BPP): ', round(bpp,2))

def PSNR(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        # The original and decompressed image are identical
        return 100
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return round(psnr, 2)

# PSNR rate-distortion curve
def rate_distortion_curve(gray_image, type, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals):
    # Create an empty array to store the PSNR values
    psnr_values = []
    x_values = []
    quantization_levels = np.arange(0.1, 1.1, 0.1)

    # Control of the compression rate
    for i in quantization_levels:
        
        print(f"Quantization level: {i}")

        # Encode the image
        bitstream, codec = encode(gray_image, quantization_matrix * i, num_blocks_height, num_blocks_width, block_size, decimals)

        # Decode the image
        decoded_image = decode(bitstream, codec, quantization_matrix * i, num_blocks_height, num_blocks_width, block_size, decimals)

        # Calculate the PSNR
        psnr = PSNR(gray_image, decoded_image)

        # Append the values
        x_values.append(len(bitstream))
        psnr_values.append(psnr)

    if type == 'bpp':
        x_values = np.array(x_values) * 8 / (gray_image.shape[0] * gray_image.shape[1])
        label = 'Bit per pixel (BPP)'
    elif type == 'scale':
        x_values = quantization_levels
        label = 'Quantization Scale'
    elif type == 'size':
        label = 'File Size (bytes)'
    else:
        raise Exception('Invalid type')
    
    fig = go.Figure(data=go.Scatter(x=x_values, y=psnr_values, mode='lines+markers', name='lines+markers', text=np.round(quantization_levels, 1)))
    fig.update_layout(title='Rate-Distortion Curve (PSNR vs. Quantization Scale)', xaxis_title=label, yaxis_title='PSNR (dB)')
    fig.show()

def display_images(original_image, compressed_image):
    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Add the original image subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original Image')

    # Add the compressed image subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(compressed_image, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Compressed Image')

    plt.show()

if __name__ == "__main__":
    
    filename = "media/input/lena1.raw"
    image_height, image_width, block_size = 256, 256, 8
    # The number of blocks in the height and width axes
    num_blocks_height = image_height // block_size
    num_blocks_width = image_width // block_size
    decimals = 0
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    quantization_matrix *= 1
    gray_image = open_raw_image(filename, image_width, image_height)
    # t = time.time()
    # encoded_image, codec = encode(gray_image, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
    # print(f"Encoding time: {round(time.time() - t, 2)}s")
    # t1 = time.time()
    # decoded_image = decode(encoded_image, codec, quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
    # print(f"Decoding time: {round(time.time() - t1, 2)}s")

    # compression_quality(gray_image, encoded_image)
    # display_images(gray_image, decoded_image)

    rate_distortion_curve(gray_image, "size", quantization_matrix, num_blocks_height, num_blocks_width, block_size, decimals)
