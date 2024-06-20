import sys
from src.logger import logging


def get_error_details(error, error_detail: sys):
    """
    Extracts detailed error information including the file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    return f"Error occurred in script: {file_name}, line: {line_number}, message: {error_message}"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = f"Error occurred in script: {self.file_name}, line: {self.line_number}, message: {error_message}"

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    logging.info("Logging has started")


