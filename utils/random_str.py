import random
import string

LETTERS = string.ascii_letters
#LETTERS = string.ascii_letters + string.digits + string.punctuation

def get_random_string(num):

    random_letters = random.choices(LETTERS, k=num)
    random_string = ''.join(random_letters)
    return random_string
