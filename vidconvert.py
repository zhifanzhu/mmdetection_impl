import argparse
from visdrone.utils import MergeTxt


def parse_args():
    parser = argparse.ArgumentParser(description='Run detector and output to txt files')
    parser.add_argument('--indir')
    parser.add_argument('--outdir')
    args = parser.parse_args()
    return args


def main()
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    MergeTxt.frames_dets2dict_then_output(indir, outdir)


if __name__ == '__main__':
    main()
