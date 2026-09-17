"""Package trained weights into submit_pt.zip with the structure required by the competition.

Usage: python package_submit.py [--design modelDesign.py] [--weights modelSubmit] [--out submit_pt.zip]

Required zip layout:
submit_pt/
├── modelDesign.py
└── modelSubmit/
    ├── transmitter.pth
    ├── receiver.pth
    └── encoder.pth
"""
import argparse
import os
import zipfile

WEIGHTS = ['transmitter.pth', 'receiver.pth', 'encoder.pth']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', default='modelDesign.py')
    parser.add_argument('--weights', default='modelSubmit')
    parser.add_argument('--out', default='submit_pt.zip')
    args = parser.parse_args()

    files = [(args.design, 'submit_pt/modelDesign.py')]
    for w in WEIGHTS:
        src = os.path.join(args.weights, w)
        if not os.path.isfile(src):
            raise SystemExit(f'missing weight file: {src}')
        size_mb = os.path.getsize(src) / 1e6
        print(f'{src}: {size_mb:.1f} MB')
        files.append((src, f'submit_pt/modelSubmit/{w}'))
    if not os.path.isfile(args.design):
        raise SystemExit(f'missing design file: {args.design}')

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
    total = os.path.getsize(args.out) / 1e6
    print(f'wrote {args.out}: {total:.1f} MB (limit 1000 MB)')
    if total > 1000:
        raise SystemExit('submission exceeds 1 GB limit!')


if __name__ == '__main__':
    main()
