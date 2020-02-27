import sys, re
from .nlptools import NLPTools

class Reference:
    def __init__(authors, title, additional, publisher, date):
        self.authors = authors
        self.title = title
        self.additional = additional
        self.publisher = publisher
        self.date = date

    @staticmethod
    def from_mla(line):
        pass

class FormatError(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Format does not match " + name

class APAFormat:
    @staticmethod
    def rate_match(line):
        if APAFormat._matchit(line):
            return 1
        else:
            return 0
    
    @staticmethod
    def _matchit(line):
        return re.match(r"(.+?)\s+\((\d{4})\)\.?\s+(.+)$", line)

    @staticmethod
    def get_authors(line):
        match = APAFormat._matchit(line)
        if match is None:
            raise FormatError("APA")
        return NLPTools.get_authors(match.group(1))

    @staticmethod
    def get_title(line):
        match = APAFormat._matchit(line)
        if match is None:
            raise FormatError("APA")
        return match.group(3).split('. ')[0]
    
    @staticmethod
    def get_journal(line):
        match = APAFormat._matchit(line)
        if match is None:
            raise FormatError("APA")
        return '. '.join(match.group(3).split('. ')[1:])
    
    @staticmethod
    def get_year(line):
        match = APAFormat._matchit(line)
        if match is None:
            raise FormatError("APA")
        return int(match.group(2))

class MLAFormat:
    @staticmethod
    def get_authors(line):
        return line.split('. ')[0].split(', ')

    @staticmethod
    def get_title(line):
        return line.split('. ')[1]
    
    @staticmethod
    def get_journal(line):
        return line.split('. ')[2]
    
    @staticmethod
    def get_year(line):
        return int(line.split('. ')[3].split(';')[0])

if __name__ == '__main__':
    with open(sys.argv[1], "r") as fp:
        for line in fp:
            #print line
            print("  " + ', '.join(APAFormat.get_authors(line)))
            #print "  " + APAFormat.get_title(line)
            #print "  " + APAFormat.get_journal(line)
            #print "  " + str(APAFormat.get_year(line))
