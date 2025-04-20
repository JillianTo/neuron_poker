import os
import csv
import re


def iter_phh_files(data_dir):
    """
    Iterate through all .phhs files under the given directory.
    """
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.phhs'):
                yield os.path.join(root, fname)


def iter_hands(file_path):
    """
    Generator that yields individual hand histories as lists of lines.
    Hands are separated by blank lines.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        hand_lines = []
        for raw in f:
            line = raw.strip()
            if not line:
                if hand_lines:
                    yield hand_lines
                    hand_lines = []
            else:
                hand_lines.append(line)
        if hand_lines:
            yield hand_lines


def parse_hand(lines):
    """
    Parse a single hand (list of strings) into a structured dict.
    Extracts header line, seat info, flop/turn/river cards, and raw actions.
    """
    hand = {
        'header': None,
        'seats': [],
        'flop': [],
        'turn': None,
        'river': None,
        'actions': []
    }
    for line in lines:
        if line.startswith('|*|') and hand['header'] is None:
            hand['header'] = line
        elif line.startswith('Seat '):
            hand['seats'].append(line)
        elif line.startswith('*** FLOP'):
            m = re.search(r'\[(.*?)\]', line)
            if m:
                hand['flop'] = m.group(1).split()
        elif line.startswith('*** TURN'):
            parts = re.findall(r'\[(.*?)\]', line)
            if len(parts) >= 2:
                hand['turn'] = parts[1]
        elif line.startswith('*** RIVER'):
            parts = re.findall(r'\[(.*?)\]', line)
            if len(parts) >= 2:
                hand['river'] = parts[1]
        else:
            hand['actions'].append(line)
    return hand


def extract_metadata_from_path(phh_file):
    """
    Derive network code and stakes string from the file path.
    """
    dirname = os.path.basename(os.path.dirname(phh_file))
    parts = dirname.split('_')
    network = parts[0].split('-')[0]
    stakes = parts[2] if len(parts) > 2 else ''
    return network, stakes


if __name__ == '__main__':
    DATA_DIR = 'data/handhq'
    OUT_FILE = 'parsed_hands.txt'
    MAX_SEATS = 10
    count = 0

    # build column headers
    base_cols = ['network', 'stakes', 'hand_id', 'sb', 'bb', 'timestamp',
                 'flop1', 'flop2', 'flop3', 'turn', 'river']
    seat_cols = []
    for i in range(1, MAX_SEATS+1):
        seat_cols += [f'seat{i}_player', f'seat{i}_stack']
    col_headers = base_cols + seat_cols + ['actions']

    with open(OUT_FILE, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f, delimiter='\t')
        writer.writerow(col_headers)

        for phh_file in iter_phh_files(DATA_DIR):
            network, stakes = extract_metadata_from_path(phh_file)
            print(f"Processing file: {os.path.basename(phh_file)} (Network: {network}, Stakes: {stakes})")

            for lines in iter_hands(phh_file):
                hand = parse_hand(lines)
                # parse header fields
                header_line = hand['header'] or ''
                m = re.search(r'Hand #(?P<id>\d+).*?\((?P<sb>\d+)/(?P<bb>\d+)\).*?[-–]\s*(?P<ts>[\d\- :]+)', header_line)
                if m:
                    hand_id, sb, bb, timestamp = m.group('id'), m.group('sb'), m.group('bb'), m.group('ts')
                else:
                    hand_id = sb = bb = timestamp = ''

                # board
                flop_cards = hand.get('flop', [])
                flop1, flop2, flop3 = (flop_cards + ['', '', ''])[:3]
                turn_card = hand.get('turn', '')
                river_card = hand.get('river', '')

                # seats
                seat_names = [''] * MAX_SEATS
                seat_stacks = [''] * MAX_SEATS
                for seat_line in hand['seats']:
                    m2 = re.match(r'Seat (\d+): (\S+) \((\d+)\)', seat_line)
                    if m2:
                        pos = int(m2.group(1)) - 1
                        if 0 <= pos < MAX_SEATS:
                            seat_names[pos] = m2.group(2)
                            seat_stacks[pos] = m2.group(3)

                # actions
                actions_str = '|'.join(hand['actions'])

                # write row
                row = [network, stakes, hand_id, sb, bb, timestamp,
                       flop1, flop2, flop3, turn_card, river_card]
                for name, stack in zip(seat_names, seat_stacks):
                    row.extend([name, stack])
                row.append(actions_str)

                writer.writerow(row)
                count += 1

                if count % 100000 == 0:
                    print(f"Parsed {count} hands so far...")

    print(f"Saved {count} hands to {OUT_FILE}")
