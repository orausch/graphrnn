"""
Experiments entrypoint
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_type', choices=['grid'])

    args = parser.parse_args()
