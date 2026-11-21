"""Start Nullpoint and open it in the browser:

    python -m nullpoint                # webcam board; a new user starts with the 10-second setup
    python -m nullpoint --sim          # no camera: arrow keys = glance, Space = blink
    python -m nullpoint --lan          # also serve the caregiver view to phones on the same Wi-Fi

All options of nullpoint.board.server work here (--mute, --no-adapt, --cam N, --port P, ...).
"""
from nullpoint.board.server import main

if __name__ == "__main__":
    main(open_browser=True)
