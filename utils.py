import os

def clean_up(elem):
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

def call_for_all_children(dirs_or_files, func, *args):
    if isinstance(dirs_or_files, str):
        dirs_or_files = (dirs_or_files,)
    for dir_or_file in dirs_or_files:
        if os.path.isdir(dir_or_file):
            for fname in os.listdir(dir_or_file):
                func(os.path.join(dir_or_file, fname), *args)
        else:
            func(dir_or_file, *args)

