"""
This file contains the main function to run the Autonomic Tester application.
"""

from src.cli.cli import main_args_parser


def main():
    """
    The main function to run the Autonomic Tester application.
    """
    args = main_args_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
