import re
import sys

def parse_diff(diff):
    lines = diff.split('\n')
    start_line_old = None
    end_line_old = None
    start_line_new = None
    end_line_new = None
    start_side = None
    side = None

    # Regular expression to match the chunk header and capture a, b, c, d
    chunk_header_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

    for i, diff_line in enumerate(lines):
        chunk_header_match = chunk_header_re.match(diff_line)
        if chunk_header_match:
            a = int(chunk_header_match.group(1))
            b = int(chunk_header_match.group(2)) if chunk_header_match.group(2) else 1
            c = int(chunk_header_match.group(3))
            d = int(chunk_header_match.group(4)) if chunk_header_match.group(4) else 1

            start_line_old = a
            end_line_old = a + b - 1
            start_line_new = c
            end_line_new = c + d - 1
            continue

        if diff_line.startswith('+') and not diff_line.startswith('+++'):
            if start_side is None:
                start_side = 'RIGHT'
            side = 'RIGHT'
        elif diff_line.startswith('-') and not diff_line.startswith('---'):
            if start_side is None:
                start_side = 'LEFT'
            side = 'LEFT'

    return start_line_new, end_line_new, start_side, side

if __name__ == "__main__":
    diff = sys.stdin.read()
    start_line, line, start_side, side = parse_diff(diff)
    print(f"{start_line} {line} {start_side} {side}")
