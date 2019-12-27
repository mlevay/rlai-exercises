from datetime import datetime


def print_status(text):
    print(text, datetime.now().strftime("%H:%M:%S"))