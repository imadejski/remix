import os

import pandas as pd

RANKED_VIEWS = [
    "PA",
    "AP",
    "LATERAL",
    "LL",
    "AP AXIAL",
    "AP LLD",
    "AP RLD",
    "PA RLD",
    "PA LLD",
    "LAO",
    "RAO",
    "LPO",
    "XTABLE LATERAL",
    "SWIMMERS",
    "",
]


def mimic_cxr_get_single_image_per_study(
    studies: list[int] | list[str],
    data_dir: str,
) -> list[str]:
    if isinstance(studies[0], str):
        try:
            int(studies[0])
            # studies are "X"
            studies = [int(s) for s in studies]
        except:
            # otherwise assume studies are in form "sX"
            studies = [int(s[1:]) for s in studies]
    df = pd.read_csv(os.path.join(data_dir, "mimic-cxr-2.0.0-metadata.csv.gz"))
    df["ViewPosition"].fillna("", inplace=True)

    mapping = {
        k: i for i, k in enumerate(RANKED_VIEWS[::-1])
    }  # higher priority gets larger number
    df["ViewRank"] = df["ViewPosition"].replace(mapping)

    df = df.sort_values(by=["study_id", "ViewRank", "StudyDate", "StudyTime"])
    unique_images = df.drop_duplicates("study_id", keep="last")

    # fmt: off
    image_paths = (
        data_dir + "/" + "jpg" + "/"
        "p" + unique_images["subject_id"].astype(str).str[:2] + "/"
        + "p" + unique_images["subject_id"].astype(str) + "/"
        + "s" + unique_images["study_id"].astype(str) + "/"
        + unique_images["dicom_id"] + ".jpg"
    )
    # fmt: on

    study_path = {s: p for s, p in zip(unique_images["study_id"], image_paths)}
    return [study_path[s] for s in studies]


def mimic_cxr_get_single_frontal_image_per_study(
    studies: list[int] | list[str],
    data_dir: str,
) -> list[str]:
    if isinstance(studies[0], str):
        try:
            int(studies[0])
            # studies are "X"
            studies = [int(s) for s in studies]
        except:
            # otherwise assume studies are in form "sX"
            studies = [int(s[1:]) for s in studies]

    df = pd.read_csv(os.path.join(data_dir, "mimic-cxr-2.0.0-metadata.csv"))
    df["ViewPosition"].fillna("", inplace=True)

    # Filter for only frontal images (PA and AP views) and rank them
    frontal_views = ["PA", "AP"]
    mapping = {
        k: i for i, k in enumerate(frontal_views[::-1])
    }  # PA gets higher rank than AP
    df = df[df["ViewPosition"].isin(frontal_views)]
    df["ViewRank"] = df["ViewPosition"].replace(mapping)

    # Sort by view rank (PA preferred), then study date and time
    df = df.sort_values(by=["study_id", "ViewRank", "StudyDate", "StudyTime"])
    unique_images = df.drop_duplicates("study_id", keep="last")

    # fmt: off
    image_paths = (
        data_dir + "/" + "jpg" + "/"
        "p" + unique_images["subject_id"].astype(str).str[:2] + "/"
        + "p" + unique_images["subject_id"].astype(str) + "/"
        + "s" + unique_images["study_id"].astype(str) + "/"
        + unique_images["dicom_id"] + ".jpg"
    )
    # fmt: on

    study_path = {s: p for s, p in zip(unique_images["study_id"], image_paths)}

    # Check if any studies are missing frontal images
    missing_studies = [s for s in studies if s not in study_path]
    assert (
        not missing_studies
    ), f"The following studies have no PA or AP images available: {missing_studies}"

    return [study_path[s] for s in studies]
