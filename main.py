from DLLSOA import DLLSOA
import utils
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="train_DLLSOA", description="load the DLLSOA config file"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="the config file path",
        default="configs/dllsoa_template.yaml",

    )

    config_arg = parser.parse_args()
    args = utils.load_config(config_arg.config_path)
    model = DLLSOA(args)
    model.train()


if __name__ == "__main__":
    main()
