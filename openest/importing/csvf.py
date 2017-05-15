import csv, re

class Reader:
    def __init__(self, iterable, delimiter=',', **kw):
        self.reader = csv.reader(iterable, delimiter=delimiter, **kw)
        self.delimiter = delimiter
        self.footnotes = {}

    def __iter__(self):
        return self

    def next(self):
        for row in self.reader:
            match = re.match(r"\[(\d+)\]\s+", row[0])
            if match is not None:
                line = self.delimiter.join(row)
                self.footnotes[match.group(1)] = line[len(match.group(0)):]
            else:
                return map(Entry, row)

class Entry:
    def __init__(self, text):
        # Look for footmarks
        footmarks = []
        while True:
            match = re.search(r"\[(\d+)\]$", text)
            if match is not None:
                footmarks.append(match.group(1))
                text = text[:-len(match.group(0))]
            else:
                break

        self.text = text
        if re.match(r"-?\d*\.?\d*%?$", text):
            if text[-1] == '%':
                self.value = float(text[:-1]) / 100.0
            else:
                self.value = float(text)
        else:
            self.value = None
            
        self.footmarks = footmarks
    
    def __str__(self):
        return self.text

    def __float__(self):
        return self.value
