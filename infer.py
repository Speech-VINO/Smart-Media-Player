import argparse
from src.predict import OpenVINOPredictor
from pprint import pprint


def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = argparse.ArgumentParser("python infer.py")
    required, optional = parser._action_groups

    required.add_argument("-i", "--input", help="Audio (.wav) file to diarize", required=True)
    required.add_argument("--xml", help="OpenVINO IR .xml file", required=True)
    required.add_argument("--bin", help="OpenVINO IR .bin file", required=True)
    required.add_argument("--config", help="model config file", required=True)
    optional.add_argument("--max-frame", help="Inference window length", default=45)
    optional.add_argument("--hop", help="Hop length of inference window", default=3)
    optional.add_argument("--plot", help="Plot the diarization result", default=True)
    args = parser.parse_args()
    return args


def infer_audio(args):
    """
    Perform speaker diarization
    """
    p = OpenVINOPredictor(args.xml, args.bin, args.config, args.max_frame, args.hop)
    timestamps, speakers = p.predict(args.input, plot=args.plot)

    result = [{"timestamp": timestamp.round(2), "speaker_id": speaker} for timestamp, speaker in zip(timestamps, speakers)]
    pprint(result)


if __name__ == "__main__":
    args = get_args()
    infer_audio(args)
