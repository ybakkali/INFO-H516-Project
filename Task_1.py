import numpy as np
import matplotlib.pyplot as plt
import time
from dahuffman import HuffmanCodec


def open_raw_image(file_path, image_width, image_height):
    # Read the raw file as binary
    with open(file_path, 'rb') as f:
        # Read the binary data
        raw_data = f.read()
    
    # Convert the raw data for 8-bit grayscale image to a numpy integer array
    image_array = np.frombuffer(raw_data, dtype=np.uint8)
    
    # Reshape the array based on the image dimensions
    image_array = image_array.reshape((image_height, image_width))
    
    return image_array

def split_image_into_blocks(image, num_blocks_width, num_blocks_height, block_size):

    # Create an empty array to store the blocks
    blocks = np.empty((num_blocks_height, num_blocks_width, block_size, block_size), dtype=np.uint8)
    # Split the image into blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            blocks[i, j] = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
    
    return blocks

def merge_blocks_into_image(image_blocks):
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_size, _ = image_blocks.shape

    # Calculate the image dimensions
    image_height = num_blocks_height * block_size
    image_width = num_blocks_width * block_size

    # Create an empty array to store the reconstructed image
    image = np.empty((image_height, image_width), dtype=np.uint8)

    # Merge the blocks into the image
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = np.clip(image_blocks[i, j], 0, 255)
    
    return image

def dct_1D(array, normalize=True):
    
    N = len(array)
    dct_coef = np.empty(N)

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
    dct_block = np.empty((M, N))

    # Compute the 1D DCT for each row
    for i in range(M):
        dct_block[i,:] = dct_1D(block[i,:])
    
    # Compute the 1D DCT for each column
    for j in range(N):
        dct_block[:,j] = dct_1D(dct_block[:,j])

    return dct_block

def idct_1D(array, normalize=True):
        
    N = len(array)
    dct_coef = np.empty(N)

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
    dct_block = np.empty((M, N))

    # Compute the 1D DCT for each row
    for i in range(M):
        dct_block[i,:] = idct_1D(block[i,:])
    
    # Compute the 1D DCT for each column
    for j in range(N):
        dct_block[:,j] = idct_1D(dct_block[:,j])

    return dct_block

def transform_coding(image_blocks, decimals=0):
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_height, block_width = image_blocks.shape

    # Create an empty array to store the DCT coefficients blocks
    dct_coefficients_blocks = np.empty((num_blocks_height, num_blocks_width, block_height, block_width))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Calculate the DCT coefficients
            
            dct_coefficients_blocks[i, j] = dct_2D(image_blocks[i, j])

    return np.round(dct_coefficients_blocks, decimals=decimals)

def inverse_transform_coding(dct_coefficients_blocks, decimals=0):
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_height, block_width = dct_coefficients_blocks.shape

    # Create an empty array to store the DCT coefficients blocks
    image_blocks = np.empty((num_blocks_height, num_blocks_width, block_height, block_width))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Calculate the DCT coefficients
            
            image_blocks[i, j] = idct_2D(dct_coefficients_blocks[i, j])

    return np.round(image_blocks, decimals=decimals)

def quantization(dct_coefficients_blocks, quantization_matrix, decimals=0):
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_height, block_width = dct_coefficients_blocks.shape

    # Create an empty array to store the quantized blocks
    quantized_blocks = np.empty((num_blocks_height, num_blocks_width, block_height, block_width))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Calculate the quantized coefficients
            quantized_blocks[i, j] = dct_coefficients_blocks[i, j] / quantization_matrix

    return np.round(quantized_blocks, decimals=decimals)

def inverse_quantization(quantized_blocks, quantization_matrix, decimals=0):
    # Get the image dimensions
    num_blocks_height, num_blocks_width, block_height, block_width = quantized_blocks.shape

    # Create an empty array to store the quantized blocks
    dct_coefficients_blocks = np.empty((num_blocks_height, num_blocks_width, block_height, block_width))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
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

    block = np.empty((block_size, block_size))

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


def zigzag_scan(quanitzed_blocks, num_blocks_height, num_blocks_width, block_size=8):

    # Create an empty array to store the zigzag scanned blocks
    zigzag_blocks = np.empty((num_blocks_height, num_blocks_width, block_size * block_size))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Calculate the zigzag scanned block
            zigzag_blocks[i, j] = zigzag_scan_block_to_array(quanitzed_blocks[i, j])

    return zigzag_blocks

def inverse_zigzag_scan(zigzag_blocks, num_blocks_height, num_blocks_width, block_size):

    # Create an empty array to reconstruct the blocks
    inverse_zigzag_blocks = np.empty((num_blocks_height, num_blocks_width, block_size, block_size))

    # Iterate over the blocks
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            # Calculate the inverse zigzag scanned block
            inverse_zigzag_blocks[i, j] = zigzag_scan_array_to_block(zigzag_blocks[i, j], block_size)

    return inverse_zigzag_blocks

def entropy_coding(zigzag_blocks):

    arr = np.array(zigzag_blocks.flatten())

    # Create a Huffman codec object
    codec = HuffmanCodec.from_data(arr)

    # Encode the data
    encoded_blocks = codec.encode(arr)

    return encoded_blocks, codec

def inverse_entropy_coding(encoded_blocks, codec, num_blocks_height, num_blocks_width, block_size):
    
    # Decode the data
    decoded_blocks = codec.decode(encoded_blocks)

    # Reshape the decoded data into the original block shape
    decoded_arr = np.array(decoded_blocks).reshape((num_blocks_height, num_blocks_width, block_size * block_size))

    return decoded_arr

def encode(image_array, quantization_matrix, num_blocks_width, num_blocks_height, block_size=8, decimals=0):

    # Split the image into blocks
    image_blocks = split_image_into_blocks(image_array, num_blocks_width, num_blocks_height, block_size)

    # Transform coding
    dct_coefficients_blocks = transform_coding(image_blocks, decimals=decimals)

    # Quantization
    quantized_blocks = quantization(dct_coefficients_blocks, quantization_matrix, decimals=decimals)

    # Zigzag scan
    zigzad_blocks = zigzag_scan(quantized_blocks, num_blocks_width, num_blocks_height)

    # Entropy coding
    encoded_blocks, codec = entropy_coding(zigzad_blocks)

    # Return the quantized blocks
    return encoded_blocks, codec

def decode(encoded_blocks, codec, quantization_matrix, num_blocks_width, num_blocks_height, block_size=8, decimals=0):

    # Inverse entropy coding
    zigzad_blocks = inverse_entropy_coding(encoded_blocks, codec, num_blocks_width, num_blocks_height, block_size)

    # Inverse zigzag scan
    quantized_blocks = inverse_zigzag_scan(zigzad_blocks, num_blocks_width, num_blocks_height, block_size)

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
    print('Compression factor: ', compression_factor)
    print('Bit per pixel (BPP): ', bpp)

def PSNR(original_image, compressed_image):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        # The original and decompressed image are identical
        return 100
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr

# (PSNR vs quantization scale)
def rate_distortion_curve(gray_image, block_size, quantization_matrix):
    # Create an empty array to store the PSNR values
    psnr_values = []

    # Iterate over the quantization matrix
    for i in range(1, 100):
        # Encode the image
        encoded_image, codec, shape = encode(gray_image, block_size, quantization_matrix * i)

        # Decode the image
        decoded_image = decode(encoded_image, codec, shape, quantization_matrix * i)

        # Calculate the PSNR
        psnr = PSNR(gray_image, decoded_image)

        # Append the PSNR value
        psnr_values.append(psnr)

    # Plot the rate-distortion curve
    plt.plot(range(1, 100), psnr_values)
    plt.xlabel('Compression Rate')
    plt.ylabel('PSNR')
    plt.show()

# (PSNR vs data size)
def rate_distortion_curve_2(gray_image, block_size, quantization_matrix):
    # Create an empty array to store the PSNR values
    psnr_values = []
    data_size = []

    # Iterate over the quantization matrix
    for i in range(1, 100):
        # Encode the image
        encoded_image, codec, shape = encode(gray_image, quantization_matrix * i, block_size)

        # Decode the image
        decoded_image = decode(encoded_image, codec, shape, quantization_matrix * i)

        # Calculate the PSNR
        psnr = PSNR(gray_image, decoded_image)

        # Append the PSNR value
        psnr_values.append(psnr)
        data_size.append(len(encoded_image) * 8)

    # Plot the rate-distortion curve
    plt.plot(data_size, psnr_values)
    plt.xlabel('Data Size')
    plt.ylabel('PSNR')
    plt.show()

def display_image(original_image, compressed_image):
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
    
    filename = "task_1/lena1.raw"
    image_width, image_height = 256, 256
    block_size = 8
    # Calculate the number of blocks in the image
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
    t = time.time()
    encoded_image, codec = encode(gray_image, quantization_matrix, num_blocks_width, num_blocks_height, block_size, decimals)
    t1 = time.time()
    print("Encoding time: ", t1 - t)
    decoded_image = decode(encoded_image, codec, quantization_matrix, num_blocks_width, num_blocks_height, block_size, decimals)
    print("Decoding time: ", time.time() - t1)

    compression_quality(gray_image, encoded_image)
    display_image(gray_image, decoded_image)

    # rate_distortion_curve(gray_image, block_size, quantization_matrix)
    # rate_distortion_curve_2(gray_image, block_size, quantization_matrix)
