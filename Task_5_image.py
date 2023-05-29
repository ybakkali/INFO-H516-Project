from Task_1 import *
from PIL import Image
from os.path import getsize

def open_raw_image_PIL(filename, image_width, image_height):
    with open(filename, 'rb') as file:
        raw_data = file.read()
    
    # Specify the appropriate mode ('L' for grayscale, 'RGB' for color, etc.)
    image = Image.frombytes('L', (image_width, image_height), raw_data)
    return image

if __name__ == "__main__":

    filename = "media/input/lena1.raw"
    image_height, image_width, block_size = 256, 256, 8
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


    gray_image = open_raw_image(filename, image_width, image_height)
    gray_image_PIL = open_raw_image_PIL(filename, image_width, image_height)



    psnr_values_task1 = []
    file_sizes_task1 = []
    # scale from 0.1 to 1.0 (best) and from 10 to 90 (worst) representing the quantization scale factor
    quantization_levels = np.concatenate((np.arange(0.1, 1.1, 0.1), [2 ** i for i in range(1, 6)]))

    # Iterate over the quantization matrix
    for i in quantization_levels:
  
        # Encode the image
        encoded_image, codec = encode(gray_image, quantization_matrix * i, num_blocks_height, num_blocks_width, block_size)

        # Decode the image
        decoded_image = decode(encoded_image, codec, quantization_matrix * i, num_blocks_height, num_blocks_width, block_size)

        # Calculate the PSNR
        psnr = PSNR(gray_image, decoded_image)

        # Append the values
        psnr_values_task1.append(psnr)
        file_sizes_task1.append(len(encoded_image))

    psnr_values_jpeg = []
    file_sizes_jpeg = []
    # scale from 0 (worst) to 95 (best)
    quality_range = range(0, 96, 10)

    for quality in quality_range:
        # Compress using JPEG
        jpeg_compressed_image = 'media/output/jpeg_compression.jpg'
        gray_image_PIL.save(jpeg_compressed_image, 'JPEG', quality=quality)
        jpeg_file_size = getsize(jpeg_compressed_image)
        psnr = PSNR(gray_image, np.array(Image.open(jpeg_compressed_image)).astype(np.uint8))
        
        psnr_values_jpeg.append(psnr)
        file_sizes_jpeg.append(jpeg_file_size)


    psnr_values_jpeg2000 = []
    file_sizes_jpeg2000 = []
    # scale from 10 to 90 representing an approximate size compression
    quality_layers_range = range(10, 100, 10)

    for layers in quality_layers_range:
        # Compress using JPEG2000
        jpeg2000_compressed_image = 'media/output/jpeg2000_compression.jp2'
        gray_image_PIL.save(jpeg2000_compressed_image, 'JPEG2000', quality_mode='rates', quality_layers=[layers], codeblock_size=(8, 8))
        jpeg2000_file_size = getsize(jpeg2000_compressed_image)
        psnr = PSNR(gray_image, np.array(Image.open(jpeg2000_compressed_image)).astype(np.uint8))
        
        psnr_values_jpeg2000.append(psnr)
        file_sizes_jpeg2000.append(jpeg2000_file_size)

    # Plot rate-distortion curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=file_sizes_task1, y=psnr_values_task1, mode='lines+markers', name='Task 1', text=np.round(quantization_levels, 1)))
    fig.add_trace(go.Scatter(x=file_sizes_jpeg, y=psnr_values_jpeg, mode='lines+markers', name='JPEG', text=list(quality_range)))
    fig.add_trace(go.Scatter(x=file_sizes_jpeg2000, y=psnr_values_jpeg2000, mode='lines+markers', name='JPEG2000', text=list(quality_layers_range)))
    fig.update_layout(title='Rate-Distortion Curve (PSNR vs. File Size)', xaxis_title='File Size (bytes)', yaxis_title='PSNR (dB)')
    fig.show()

    # display_images(original_image, Image.open(jpeg_compressed_image))
    # display_images(original_image, Image.open(jpeg2000_compressed_image))

