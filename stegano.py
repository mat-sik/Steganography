""" 
Mateusz Sikorski
zrealizowałem zadania
1, 2, 3, 4, 5, 6, 7
"""
import numpy as np
from PIL import Image
import argparse


def read_image_greyscale(file):
    """ Loads image, transforms it to grayscale, returns shape and data of the image. """
    im = Image.open(file)
    im_con = Image.open(file).convert('LA')
    data = im_con.getdata()
    data = np.array(data, dtype=np.uint8)
    return im.size, np.array(data, dtype=np.uint8)


def read_image(file):
    """ Reads image without turning it to grayscale. """
    im = Image.open(file)
    data = im.getdata()
    return im.size, np.array(data, dtype=np.uint8)


def write_image_greyscale(x, y, bit_vector, output_image_file):
    """ Saves image to grayscale. """
    bit_vector = bit_vector.reshape(y, x, 2)
    img = Image.fromarray(bit_vector, "LA")
    img.save(f'{output_image_file}.png')
    # img.show()


def write_image(x, y, bit_vector, output_image_file):
    """ Saves image(RGB) in which we hide image(greyscale). """
    bit_vector = bit_vector.reshape(y, x, 3)
    img = Image.fromarray(bit_vector, "RGB")
    img.save(f'{output_image_file}.png')
    # img.show()


def read_DNA(text_file):
    """ Reads dna from fasta then encodes it. """
    with open(text_file, 'r') as f:
        contents = ''.join(f.read().splitlines()[1:-1])
    coding = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    code = ''
    for i in contents:
        code += coding[i]
    return code


def write_DNA(bit_vector, output_text_file):
    """ Decodes and saves to output_... """
    coding = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
    contents = ''
    for el in [f'{bit_vector[i]}{bit_vector[i+1]}' for i in range(0, len(bit_vector), 2)]:
        contents += coding[el]
    with open(output_text_file, 'w') as f:
        f.write(contents)


def binaryToDecimal(n):
    """ Binary number to decimal. """
    num = 0
    for t, i in enumerate(reversed(n)):
        if i == '1':
            num += 2**t 
    return num


def encoding(matrix, bit_vector):
    """ Checks whether Image is big enough and if so encodes it with bit_vector. """
    if len(bit_vector) > len(matrix):
        raise Exception("Image is too small")
    # we inverse the vector so that we will be able to write of the end of it
    matrix = matrix[::-1]
    # Encoding with boundary conditions
    for i in range(len(bit_vector)):
        if matrix[i] % 2 != int(bit_vector[i]):
            if matrix[i] == 255:
                matrix[i] -= 1
            elif matrix[i] == 0:
                matrix[i] += 1
            else:
                matrix[i] += 1
    # inverting again
    return matrix[::-1]


def encode_DNA(bit_vector, image_file, output_file):
    """
    Hides DNA code in Image.
    headline - string of bits which corresponds to the type of information which is being hidden
    and length of the information. Later we insert headline into bit_vector.
    """
    headline = '0' + np.binary_repr(len(bit_vector), width=32)
    bit_vector = headline + bit_vector
    size, M = read_image(image_file)
    # Shape of the matrix is changed to a single vector
    M = M.reshape(size[1]*size[0]*3)
    M = encoding(M, bit_vector)
    write_image(size[0], size[1], M, output_file)


def encode_image(bit_vector, x, y, image_file, output_file):
    """ Hides grayscale image inside another image. """
    headline = "1" + np.binary_repr(x, width=16) + np.binary_repr(y, width=16)
    size, M = read_image(image_file)
    M = M.reshape(size[1]*size[0]*3)
    # Shape of the matrix is changed to a single vector
    bit_vector = bit_vector.reshape(x*y*2)
    # Every decimal value is changed to binary
    bit_vector = np.array([np.binary_repr(i, width=8) for i in bit_vector])
    bit_vector = headline + ''.join(bit_vector)
    M = encoding(M, bit_vector)
    write_image(size[0], size[1], M, output_file)


def decode(image_file, output_file):
    """ Decodes image. """
    size, M = read_image(image_file)
    x, y = size
    M = M.reshape(x*y*3)
    M = M[::-1]
    # first 33 bits correspond to headline
    headline = M[0:33]
    # decoding headline
    decoded_headline = ''
    for i in headline:
        if i % 2 == 0:
            decoded_headline += '0'
        else:
            decoded_headline += '1'
    # fist bit determines what kind of information is encoded
    if decoded_headline[0] == '0':  # DNA
        # the rest of 32 bits corresponds to the length of the information
        dna_code_length = binaryToDecimal(decoded_headline[1:])
        hidden_dna_code = ''
        for i in M[33: dna_code_length+33]:
            if i % 2 == 0:
                hidden_dna_code += '0'
            else:
                hidden_dna_code += '1'
        write_DNA(hidden_dna_code, f'{output_file}')

    elif decoded_headline[0] == '1':  # IMAGE
        # two times 16 bits which correspond to x, y shape of the image
        x, y = binaryToDecimal(decoded_headline[1:17]), binaryToDecimal(decoded_headline[17:])
        hidden_pic_length = x*y
        hidden_pic_bits = ''
        # width times eight times two because that's how many bits describes out image
        for i in M[33: (2*8*hidden_pic_length)+33]:
            if i % 2 == 0:
                hidden_pic_bits += '0'
            else:
                hidden_pic_bits += '1'
        # decoded bits are being connected into eights - np.uint8
        hidden_pic_bits = [hidden_pic_bits[i:i+8] for i in range(0, len(hidden_pic_bits), 8)]
        # 8bits are changed into decimal numbers
        hidden_pic_values = np.array([binaryToDecimal(i) for i in hidden_pic_bits], dtype=np.uint8)
        write_image_greyscale(x, y, hidden_pic_values, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Choose IMAGE mode", action='store_true')
    parser.add_argument("-d", "--DNA", help="Choose DNA mode", action='store_true')
    parser.add_argument("-dec", "--decode", help="Decode IMAGE", action='store_true')
    parser.add_argument("-p", "--path", help="Give path of IMAGE|DNA to hide|decode", type=str)
    parser.add_argument("-pm", "--path_main", help="Give path of IMAGE where to hide", type=str)
    parser.add_argument("-n", "--name", help="Give name of new transformed to grayscale IMAGE", type=str)
    parser.add_argument("-nz", "--name_encoded", help="Give name of encoded IMAGE|decoded IMAGE", type=str)
    args = parser.parse_args()

    if args.image:
        size, data = read_image_greyscale(args.path)
        x, y = size
        write_image_greyscale(x, y, data, args.name)
        size, data = read_image_greyscale(f'{args.name}.png')
        encode_image(data, x, y, args.path_main, args.name_encoded)
    elif args.DNA:
        data = read_DNA(args.path)
        encode_DNA(data, args.path_main, args.name_encoded)
    elif args.decode:
        decode(args.path, args.name_encoded)


if __name__ == '__main__':
    main()
