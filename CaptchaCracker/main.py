from readdata import read_training_data
from readdata import read_data
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import sys


def threshold_filter(image):
    """ Sets all pixel with a value above threshold to 255 """
    threshold = 100

    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] > threshold:
                image[y][x] = 255


def is_column_empty(image, column):
    """Returns true, if the column is empty"""
    for y in range(len(image)):
        if image[y][column] < 255:
            return False
    return True


def find_y_bounds(image):
    """Returns the y bounds of the image"""
    lower_bound = -1
    upper_bound = -1
    for y in range(len(image)):
        row_empty = True
        for pixel in image[y]:
            if pixel < 255:
                row_empty = False
                break
        if not row_empty and lower_bound == -1:
            lower_bound = y
        if row_empty and lower_bound != -1 and upper_bound == -1:
            upper_bound = y

    return lower_bound, upper_bound


def extract_characters(image):
    """ Extracts the characters from the image using the functions defined above """
    # filter image
    threshold_filter(image)

    # find the y bounds
    lower_y, upper_y = find_y_bounds(image)

    character_starts = []
    in_character = False
    for x in range(len(image[0])):
        if is_column_empty(image, x):
            if in_character:
                in_character = False

        else:
            if not in_character:
                in_character = True
                character_starts.append(x)

    np_image = np.array(image)
    characters = []
    for character_start in character_starts:
        characters.append(np_image[lower_y:upper_y, character_start:character_start+8])

    return characters


def prepare_training_set():
    """ Loads the training set and prepares the data """
    raw_training_data = read_training_data()
    training_set = []
    training_solution_set = []

    # iterate through all training example
    for example in raw_training_data:
        image = example[0]
        solution = example[1]
        characters = extract_characters(image)

        for i in range(len(characters)):
            # only add classes we have not yet had an example before. A little hacky, but minimizes model size
            if not solution[i] in training_solution_set:
                training_set.append(characters[i].ravel())
                training_solution_set.append(solution[i])

    clf = SVC(C=100, gamma=0.0001)
    clf.fit(training_set, training_solution_set)
    joblib.dump(clf, 'model.pkl', compress=9)


def main():
    # uncomment this to create new model
    prepare_training_set()
    # return
    # load model back
    clf = joblib.load('model.pkl')

    # read sample
    image = read_data(sys.stdin)

    # extract the characters
    characters = extract_characters(image)

    for character in characters:
        prediction = clf.predict(character.ravel().reshape(1, -1))
        sys.stdout.write(prediction[0])

    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
