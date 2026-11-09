
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "SIDD_Small_sRGB_Only"
    / "Data"
)


def verify_dataset():

    print("=" * 65)
    print("HYBRIDISP — SIDD DATASET VERIFICATION")
    print("=" * 65)

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset folder not found.")
        print(f"Expected location:\n{DATASET_DIR}")
        print("\nExtract the dataset into the expected folder.")
        return

    scene_dirs = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_dir()
    )

    print(f"\nScene folders: {len(scene_dirs)}")

    total_noisy = 0
    total_gt = 0
    valid_pairs = 0
    missing_pairs = 0
    dimension_errors = 0
    unreadable_files = 0

    for scene in scene_dirs:

        noisy_files = sorted(
            scene.glob("NOISY_SRGB_*.PNG")
        )

        gt_files = sorted(
            scene.glob("GT_SRGB_*.PNG")
        )

        total_noisy += len(noisy_files)
        total_gt += len(gt_files)

        for noisy in noisy_files:

            suffix = noisy.name.replace(
                "NOISY_SRGB_", ""
            )

            gt = scene / f"GT_SRGB_{suffix}"

            if not gt.exists():
                missing_pairs += 1
                print(f"MISSING GT: {noisy}")
                continue

            try:
                with Image.open(noisy) as noisy_img:
                    noisy_size = noisy_img.size

                with Image.open(gt) as gt_img:
                    gt_size = gt_img.size

                if noisy_size != gt_size:
                    dimension_errors += 1
                    print(
                        f"DIMENSION MISMATCH: {noisy.name} "
                        f"{noisy_size} vs {gt_size}"
                    )
                    continue

                valid_pairs += 1

            except Exception as error:
                unreadable_files += 1
                print(f"UNREADABLE: {noisy.name}: {error}")

    print("\n" + "-" * 65)
    print("VERIFICATION SUMMARY")
    print("-" * 65)

    print(f"Noisy images       : {total_noisy}")
    print(f"Ground-truth images: {total_gt}")
    print(f"Valid image pairs  : {valid_pairs}")
    print(f"Missing pairs      : {missing_pairs}")
    print(f"Dimension errors   : {dimension_errors}")
    print(f"Unreadable files   : {unreadable_files}")

    if valid_pairs > 0 and missing_pairs == 0 and dimension_errors == 0:
        print("\nRESULT: Dataset pairing checks passed.")
        print("You can proceed to evaluation.")
    else:
        print("\nRESULT: Dataset needs inspection before evaluation.")

    print("=" * 65)


if __name__ == "__main__":
    verify_dataset()
