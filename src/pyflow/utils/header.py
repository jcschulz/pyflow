import textwrap

def write_header(s, name):
    '''Prints a section header.'''
    underline = '-'*80
    s.write(f'\n{name}\n{underline}\n')

def wrap(s, indent=-1, initial_indent=-1, width=-1):
    '''Format a string of list of strings into a paragraph.'''

    if indent < 0:
        indent = 0

    if initial_indent < 0:
        initial_indent = indent

    if width < 0:
        width = 80

    if not isinstance(s,str):
        s = ' '.join(s)

    s = textwrap.fill(s.strip(),
        width=80,
        initial_indent = ' '*indent,
        subsequent_indent = ' '*indent,
        replace_whitespace = True,
    )
    s = ' '*initial_indent + s.lstrip()
    return s