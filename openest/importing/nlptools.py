import re

class NLPTools(object):
    # XXX: This should understand all of the following:
    # James Rising
    # Rising, J.
    # James Rising and Fred Flintstone
    # James Rising, Fred Flintstone
    # Rising, James and Flintstone, Fred
    # Rising, J. and Flintstone, F.
    # James Rising, Fred Flintstone<,> and Solomon Hsiang
    # Rising, J., Flintstone, F.<,> and Hsiang, S.
    # Also, should handle common multi-word names like Von Thrumple
    @staticmethod
    def get_authors(authors):
        """Turns a string of authors into a canonical list.

        ['Flintstone, Fred', 'Rising, James']
        """
        alist = None
        if re.match('(.+? and )+.+', authors):
            alist = authors.split(' and ')
        if alist is None and re.match('^([^,]+?\s+[^,]+?, )+.+', authors):
            alist = authors.split(', ')
            if alist[-1][0:4] == 'and ':
                alist[-1] = alist[-1][4:]
        if alist is None and " and " in authors:
            alist = authors.split(" and ")
        if alist is None:
            alist = [authors]
        
        for ii in range(len(alist)):
            if ', ' not in alist[ii] and ' ' in alist[ii]:
                alist[ii] = alist[ii].split(' ')[-1] + ", " + ' '.join(alist[ii].split(' ')[0:-1])

        alist = [a[0:-1] if a[-1] == ',' else a for a in alist]

        return alist

    @staticmethod
    def get_authors_abbr(authors):
        if re.search('et\.? al\.?', authors):
            return authors

        alist = NLPTools.get_authors(authors)
        print(alist)
        if len(alist) > 2:
            return alist[0].split(', ')[0] + ' et al.'
        elif len(alist) > 1:
            return ', '.join([name.split(', ')[0] for name in alist[0:-1]]) + ' and ' + alist[-1].split(', ')[0]
        else:
            return alist[0].split(', ')[0]
