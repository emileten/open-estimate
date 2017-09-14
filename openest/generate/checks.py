import re

PATTERN_NONWORD = re.compile(r"\W")

def loosematch(one, two):
    one = PATTERN_NONWORD.sub("", one).lower()
    two = PATTERN_NONWORD.sub("", two).lower()

    return one == two

if __name__ == '__main__':
    print loosematch("1,000 ", "1000")
    print loosematch("1 frog", "1 toad")
    
