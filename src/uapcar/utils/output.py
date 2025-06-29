
def print_progress(i, total):
    """Prints a progress bar, starting at the beginning of the current line."""

    progress_rel = i / total
    progress_blocks = (int)((progress_rel + 0.01) * 50)
    print('\r[' + progress_blocks * '#' + (50 - progress_blocks) * ' ' + '] (' + '{:06.2f}'.format(progress_rel * 100) + ' %)', end='')
