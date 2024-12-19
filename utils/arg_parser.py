import argparse


def get_argument_parser():
    """
    Creates and configures an argument parser for the application.

    Returns:
        argparse.ArgumentParser: Configured parser with necessary arguments.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="A versatile tool for reading and solving graph-related problems."
    )

    parser.add_argument(
        '-i', '--input', required=True, type=str,
        help="Path to the input file containing the graph data. Supports image formats."
    )

    parser.add_argument(
        '-a', '--algorithm', required=True, type=str,
        help="The algorithm to be applied"
    )

    parser.add_argument(
        '-v', '--variant', type=str, default=None,
        help="Specific variant of the chosen algorithm (if applicable)."
    )

    return parser
