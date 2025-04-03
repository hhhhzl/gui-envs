#!/usr/bin/python
import argparse
from src.recording.gui import Interface
from src.utils import abspath

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Screen Recording.')
    parser.add_argument('--o', type=str, default='both',
                        help='Recording Option: both, screen, keyboard.')
    parser.add_argument('--a', type=str, default='both',
                        help='Recording Option: both, screen, keyboard.')
    parser.add_argument('--d', type=str, default=None,
                        help='Recording Domain')
    parser.add_argument('--td', type=str, default=None,
                        help='Task Description')
    parser.add_argument('--path', type=str, default=abspath('data'),
                        help='Path to Store Data')
    args = parser.parse_args()

    Interface(
        domain=args.d,
        task_description=args.td,
        app=args.a,
        mode=args.o,
        path=args.path
    ).start()
