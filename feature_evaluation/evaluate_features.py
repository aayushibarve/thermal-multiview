import cv2
import numpy as np
import pandas as pd
import tifffile
import itertools
import random
import os
import glob
import warnings
warnings.filterwarnings("ignore")

DATASET_DIR         = "/home/aayushi/Documents/Lepton/turntable"   
IMAGE_GLOB          = "*.tiff"
CALIBRATION_PATH    = "/home/aayushi/semester-project/camera_params.npz"
N_PAIRS             = 100         
MAX_FEATURES        = 500
MATCH_RATIO         = 0.75
MIN_MATCH_COUNT     = 5
ESSENTIAL_THRESHOLD = 1.0
RANDOM_SEED         = 42

DETECTORS = ['sift', 'orb', 'akaze', 'kaze', 'brisk', 'fast_brief', 'fast_freak']

def preprocess_linear(img_raw):
    norm = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

def preprocess_log(img_raw):
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY) if img_raw.ndim == 3 else img_raw
    gray = gray.astype(np.float32) + 1.0
    log  = np.log(gray)
    norm = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm

def preprocess_histeq(img_raw):
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY) if img_raw.ndim == 3 else img_raw
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.equalizeHist(norm)

def preprocess_clahe(img_raw):
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY) if img_raw.ndim == 3 else img_raw
    norm  = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(norm)

def preprocess_gauss_sharpen(img_raw):
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY) if img_raw.ndim == 3 else img_raw
    norm     = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred  = cv2.GaussianBlur(norm, (5, 5), 1.0)
    sharpened = cv2.addWeighted(norm, 1.8, blurred, -0.8, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def preprocess_gamma(img_raw, gamma=0.5):
    gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY) if img_raw.ndim == 3 else img_raw
    norm  = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(norm, table)

PREPROCESSORS = {
    "Linear"         : preprocess_linear,
    "Log Norm"       : preprocess_log,
    "Hist Eq"        : preprocess_histeq,
    "CLAHE"          : preprocess_clahe,
    "Gauss+Sharpen"  : preprocess_gauss_sharpen,
    "Gamma(0.5)"     : preprocess_gamma,
}

def extract(gray_img, method):
    m = method.lower()
    kp, desc = [], None

    try:
        if m == 'sift':
            det = cv2.SIFT_create(nfeatures=MAX_FEATURES)
            kp, desc = det.detectAndCompute(gray_img, None)

        elif m == 'orb':
            det = cv2.ORB_create(nfeatures=MAX_FEATURES, scaleFactor=1.2, nlevels=8)
            kp, desc = det.detectAndCompute(gray_img, None)

        elif m == 'akaze':
            det = cv2.AKAZE_create()
            kp, desc = det.detectAndCompute(gray_img, None)

        elif m == 'kaze':
            det = cv2.KAZE_create()
            kp, desc = det.detectAndCompute(gray_img, None)

        elif m == 'brisk':
            det = cv2.BRISK_create()
            kp, desc = det.detectAndCompute(gray_img, None)

        elif m == 'fast_brief':
            fast  = cv2.FastFeatureDetector_create(threshold=25)
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp    = fast.detect(gray_img, None)
            kp, desc = brief.compute(gray_img, kp)

        elif m == 'fast_freak':
            fast  = cv2.FastFeatureDetector_create(threshold=25)
            freak = cv2.xfeatures2d.FREAK_create()
            kp    = fast.detect(gray_img, None)
            kp, desc = freak.compute(gray_img, kp)

        else:
            raise ValueError(f"Unknown detector: {method}")

    except (cv2.error, AttributeError) as e:
        return [], None

    if kp and len(kp) > MAX_FEATURES:
        kp = sorted(kp, key=lambda x: -x.response)[:MAX_FEATURES]
        try:
            if m in ('sift', 'orb', 'akaze', 'kaze', 'brisk'):
                det_obj = {
                    'sift': cv2.SIFT_create, 'orb': cv2.ORB_create,
                    'akaze': cv2.AKAZE_create, 'kaze': cv2.KAZE_create,
                    'brisk': cv2.BRISK_create,
                }.get(m)
                if det_obj:
                    kp, desc = det_obj().compute(gray_img, kp)
        except Exception:
            pass

    return kp if kp else [], desc

def get_matcher(method):
    float_desc = {'sift', 'kaze'}
    norm = cv2.NORM_L2 if method.lower() in float_desc else cv2.NORM_HAMMING
    return cv2.BFMatcher(norm, crossCheck=False)

def match_ratio_test(desc1, desc2, matcher):
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []
    try:
        raw = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    good = [m for m, n in raw if m.distance < MATCH_RATIO * n.distance]
    return good

def load_calibration(npz_path):
    cal  = np.load(npz_path)
    K    = cal['K'].astype(np.float64)
    dist = cal['dist'].astype(np.float64)
    return K, dist

def compute_precision_recall(kp1, kp2, matches, K, dist):
    total_m  = len(matches)
    total_kp = min(len(kp1), len(kp2))

    if total_m < MIN_MATCH_COUNT:
        return 0.0, 0.0, 0, total_m

    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    pts1_u = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist, P=K)
    pts2_u = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist, P=K)
    pts1_u = pts1_u.reshape(-1, 2)
    pts2_u = pts2_u.reshape(-1, 2)

    E, mask_E = cv2.findEssentialMat(
        pts1_u, pts2_u, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ESSENTIAL_THRESHOLD
    )
    if E is None or mask_E is None:
        return 0.0, 0.0, 0, total_m

    n_cheirality, _, _, _ = cv2.recoverPose(E, pts1_u, pts2_u, K, mask=mask_E.copy())

    inliers   = int(n_cheirality)
    precision = inliers / total_m  if total_m  > 0 else 0.0
    recall    = inliers / total_kp if total_kp > 0 else 0.0
    return precision, recall, inliers, total_m

def load_images(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder} matching {pattern}")
    images = {}
    for p in paths:
        name = os.path.basename(p)
        try:
            raw = tifffile.imread(p)
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            elif raw.shape[2] == 4:
                raw = raw[:, :, :3]
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            images[name] = raw
        except Exception as e:
            print(f"Could not load {name}: {e}")
    print(f"Loaded {len(images)} image(s) from {folder}")
    return images

def build_count_table(images):
    cols = pd.MultiIndex.from_tuples(
        [(prep, det) for prep in PREPROCESSORS for det in DETECTORS],
        names=["Preprocessing", "Detector"]
    )
    df = pd.DataFrame(index=list(images.keys()), columns=cols, dtype=int)

    for img_name, raw in images.items():
        for prep_name, prep_fn in PREPROCESSORS.items():
            gray = prep_fn(raw)
            for det in DETECTORS:
                try:
                    kp, _ = extract(gray, det)
                    df.loc[img_name, (prep_name, det)] = len(kp) if kp else 0
                except Exception:
                    df.loc[img_name, (prep_name, det)] = 0

    df.loc["TOTAL"] = df.drop("TOTAL", errors="ignore").sum()
    return df

def eval_pairs(images, n_pairs):
    K, dist = load_calibration(CALIBRATION_PATH)
    names = list(images.keys())
    all_pairs = list(itertools.combinations(names, 2))

    random.seed(RANDOM_SEED)
    if len(all_pairs) <= n_pairs:
        chosen = all_pairs
        print(f"  Only {len(all_pairs)} unique pairs available — using all.")
    else:
        chosen = random.sample(all_pairs, n_pairs)

    records = []
    for (n1, n2) in chosen:
        raw1, raw2 = images[n1], images[n2]
        for prep_name, prep_fn in PREPROCESSORS.items():
            g1 = prep_fn(raw1)
            g2 = prep_fn(raw2)
            for det in DETECTORS:
                try:
                    kp1, d1 = extract(g1, det)
                    kp2, d2 = extract(g2, det)
                    matcher  = get_matcher(det)
                    matches  = match_ratio_test(d1, d2, matcher)
                    prec, rec, inl, total_m = compute_precision_recall(kp1, kp2, matches, K, dist)
                    records.append({
                        "Image 1"       : n1,
                        "Image 2"       : n2,
                        "Preprocessing" : prep_name,
                        "Detector"      : det.upper(),
                        "KP1"           : len(kp1) if kp1 else 0,
                        "KP2"           : len(kp2) if kp2 else 0,
                        "Matches"       : total_m,
                        "Inliers"       : inl,
                        "Precision"     : round(prec, 3),
                        "Recall"        : round(rec, 3),
                    })
                except Exception as e:
                    records.append({
                        "Image 1": n1, "Image 2": n2,
                        "Preprocessing": prep_name, "Detector": det.upper(),
                        "KP1": 0, "KP2": 0, "Matches": 0, "Inliers": 0,
                        "Precision": 0.0, "Recall": 0.0,
                    })

    return pd.DataFrame(records)

def summarize_pr(pr_df):
    return (
        pr_df.groupby(["Preprocessing", "Detector"])[["Precision", "Recall"]]
        .mean().round(3)
        .sort_values("Precision", ascending=False)
    )

def main():
    images = load_images(DATASET_DIR, IMAGE_GLOB)
    count_df = build_count_table(images)
    print(f"Evaluating {N_PAIRS} random pairs")
    pr_df      = eval_pairs(images, N_PAIRS)
    summary_df = summarize_pr(pr_df)
    out_dir = "/home/aayushi/semester-project/turntable_detector_results"
    os.makedirs(out_dir, exist_ok=True)
    count_path   = os.path.join(out_dir, "feature_counts.csv")
    pr_path      = os.path.join(out_dir, "precision_recall_pairs.csv")
    summary_path = os.path.join(out_dir, "pr_summary.csv")

    count_df.to_csv(count_path)
    print(f"Feature count table - {count_path}")

    if not pr_df.empty:
        pr_df.to_csv(pr_path, index=False)
        summary_df.to_csv(summary_path)
        print(f"Per-pair P/R details  - {pr_path}")
        print(f"P/R summary           - {summary_path}")

    count_no_total = count_df.drop("TOTAL", errors="ignore")
    avg_counts = count_no_total.mean(axis=0)
    avg_counts.index = [f"{prep} + {det}" for prep, det in avg_counts.index]
    top10_counts = avg_counts.sort_values(ascending=False).head(10)
    print("\nTop 10 preprocessing+detector pairs by avg features detected per image:")
    for rank, (name, val) in enumerate(top10_counts.items(), 1):
        print(f"  {rank:>2}. {name:<35} {val:.1f}")

    if not summary_df.empty:
        top5_prec = summary_df.sort_values("Precision", ascending=False).head(5)
        print("\nTop 5 by Precision:")
        for rank, ((prep, det), row) in enumerate(top5_prec.iterrows(), 1):
            print(f"  {rank}. {prep} + {det:<12} Precision={row['Precision']:.3f}  Recall={row['Recall']:.3f}")

        top5_rec = summary_df.sort_values("Recall", ascending=False).head(5)
        print("\nTop 5 by Recall:")
        for rank, ((prep, det), row) in enumerate(top5_rec.iterrows(), 1):
            print(f"  {rank}. {prep} + {det:<12} Precision={row['Precision']:.3f}  Recall={row['Recall']:.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()