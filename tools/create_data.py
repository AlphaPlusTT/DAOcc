import argparse
from os import path as osp

from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database
from data_converter import kitti_converter as kitti


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )

def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    # create_groundtruth_database(
    #     'WaymoDataset',
    #     out_dir,
    #     info_prefix,
    #     f'{out_dir}/{info_prefix}_infos_train.pkl',
    #     relative_path=False,
    #     with_mask=False)


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
