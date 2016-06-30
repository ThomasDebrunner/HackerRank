import glob, os


def read_training_data():
    n_training_samples = 25
    training_samples = []
    for i in range(n_training_samples):
        with open('sampleCaptchas/input/input%02d.txt' % i, 'r') as f:
            image = read_data(f)
        with open('sampleCaptchas/output/output%02d.txt' % i) as f:
            solution = next(f)
        training_samples.append((image, solution))
    return training_samples


def read_data(file):
    # read width and height
    height, width = [int(x) for x in next(file).split()]

    # prepare image buffer
    image_buffer = []

    # read the pixels line by line
    for line in file:
        pixel_strs = [x for x in line.split()]

        if len(pixel_strs) != width:
            raise ValueError("Incorrect length")

        line_buffer = []

        for pixel_str in pixel_strs:
            pvalues = ([int(x) for x in pixel_str.split(',')])
            gray_value = int(round(0.1140 * pvalues[0] + 0.5870 * pvalues[1] + 0.2989 * pvalues[2]))

            line_buffer.append(gray_value)

        image_buffer.append(line_buffer)

    return image_buffer
