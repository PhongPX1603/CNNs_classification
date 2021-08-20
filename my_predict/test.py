import argparse
import os
import sys
from pathlib import Path

import cv2
from natsort import natsorted

sys.path.append(os.environ['PWD'])

import utils  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image dir.')
    parser.add_argument('--output-dir', type=str, help='path to save image')
    parser.add_argument('--pattern', help='glob pattern if image_path is a dir.')
    parser.add_argument('--show-image', action='store_true')
    parser.add_argument('--start-index', default=1)
    args = parser.parse_args()

    image_paths = list(Path(args.image_path).glob(args.pattern)) if args.pattern else [Path(args.image_path)]
    image_paths = natsorted(image_paths, key=lambda x: x.stem) 

    config = utils.load_yaml(args.outputs_dir)
    predictor = utils.create_instance(config['cifar10'])

    for idx, image_path in enumerate(image_paths[int(args.start_index) - 1:], int(args.start_index)):
        print('-' * 50)
        print(f'{idx} / {len(image_paths)} - {image_path}')

        images = [cv2.imread(str(image_path))]

        outputs = predictor(images=images)
        for i, (class_name, class_score) in enumerate(outputs):
            print(f'[..] RESULT #{i + 1}: {class_name} - {class_score * 100:.2f}%')

