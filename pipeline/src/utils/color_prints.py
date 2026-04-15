#
# Printing with colors 
#

class bcolors:
    # styles
    NORMAL = ''
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'

    # reset
    ENDC = '\033[0m'

    # standard foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # aliases matching your original names
    HEADER = BRIGHT_MAGENTA
    OKBLUE = BRIGHT_BLUE
    OKCYAN = BRIGHT_CYAN
    OKGREEN = BRIGHT_GREEN
    WARNING = BRIGHT_YELLOW
    FAIL = BRIGHT_RED

    # background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'

    # extra 256-color examples
    ORANGE = '\033[38;5;208m'
    PINK = '\033[38;5;213m'
    PURPLE = '\033[38;5;129m'
    TEAL = '\033[38;5;51m'
    GRAY = '\033[38;5;245m'

LOG_STYLES = {
    "debug":   (bcolors.YELLOW,   bcolors.NORMAL),
    "info":    (bcolors.CYAN,   bcolors.NORMAL),
    "success": (bcolors.GREEN,  bcolors.NORMAL),
    "warning": (bcolors.ORANGE, bcolors.BOLD),
    "error":   (bcolors.RED,    bcolors.BOLD),
    "critical":(bcolors.WHITE + bcolors.BG_RED, bcolors.BOLD),
    "condition_check":(bcolors.MAGENTA, bcolors.ITALIC),
    "entering_function":(bcolors.BRIGHT_GREEN, bcolors.ITALIC),
}

def cprint(text: str, color: str = bcolors.NORMAL, style: str = bcolors.NORMAL) -> None:
    """A printing util function which adds colors and style
        
    Args:
        text (str): text to print
        color (_type_, optional): Color of the text(choose from bcolors class). Defaults to bcolors.NORMAL:str.
        style (_type_, optional): Style:(bold, italic etc...). Defaults to bcolors.NORMAL:str.
    """
    print(color, style)
    print(f"{style}{color}{text}{bcolors.ENDC}")
    