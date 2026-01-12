"""
This file contains the main function to run the Autonomic Tester application.
"""

from src.cli.cli import main_args_parser
import logging, sys

def main():
    """
    The main function to run the Autonomic Tester application.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = main_args_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
