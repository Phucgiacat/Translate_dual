import io
import re
import sys

if __name__ == "__main__":
    file_path = sys.argv[1]
    file_out = sys.argv[2]
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f_out = open(file_out, 'w', encoding='utf-8')
    length = len(lines)
    print(length)
    for idx, line in enumerate(lines):
        line_ = line
        parts = line.split()
        for i, part in enumerate(parts):
            if not part.isdigit():
                part = re.sub(r'\d+', '', part)
                parts[i] = part
            if part == '-' and parts[i - 1] == ":polarity":
                parts[i] = 'not'
            if part[-1] == '-' and len(part) > 1:
                parts[i] = part[:-1]
        line = ' '.join(p for p in parts)
        # res = read_amr(line)
        # print(idx)
        # print(line)
        if idx + 1 == length:
            f_out.write(line)
        else:
            f_out.write(line + '\n')

    print(f_out.seek(0, io.SEEK_END))
    f_out.close()
