import re
r = "(hi|hello|hey)[]*([a-z]*)"
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)
re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
re.match(r, "hey, what's up", flags=re.IGNORECASE)
