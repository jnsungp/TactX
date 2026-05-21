import os
import re
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


PAIR_SENSORS = {
    "de_pair": ["daimon", "eflesh"],
    "ef_pair": ["eflesh", "flexi"],
    "fd_pair": ["flexi", "daimon"],
}


def parse_ep_fr(path):
    name = os.path.basename(path)
    m = re.match(r"ep(\d+)_fr(\d+)\.pt", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def load_tensor(path):
    x = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(x, dict):
        for k in ["x", "data", "tactile", "obs"]:
            if k in x:
                x = x[k]
                break

    return torch.as_tensor(x).float()


def minmax_norm(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)


def get_available_episodes(sensor_dir):
    files = glob.glob(os.path.join(sensor_dir, "ep*_fr*.pt"))
    eps = set()

    for f in files:
        parsed = parse_ep_fr(f)
        if parsed is None:
            continue
        ep, fr = parsed
        eps.add(ep)

    return sorted(eps)


def frame_score_sequence(sensor_dir, episode, baseline_n=1):
    files = sorted(glob.glob(os.path.join(sensor_dir, f"ep{episode:04d}_fr*.pt")))

    parsed = []
    for f in files:
        p = parse_ep_fr(f)
        if p is not None:
            parsed.append((p, f))

    parsed = sorted(parsed, key=lambda x: x[0][1])

    if len(parsed) == 0:
        return None, None

    frames = np.array([p[1] for p, _ in parsed])
    xs = [load_tensor(f) for _, f in parsed]

    # baseline: first baseline_n frames average
    baseline_n = max(1, min(baseline_n, len(xs)))
    x0 = torch.stack(xs[:baseline_n], dim=0).mean(dim=0)

    scores = []
    for x in xs:
        score = (x - x0).abs().mean().item()
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)
    return frames, scores


def plot_one_episode(root, pair, obj, episode, out_dir, baseline_n=1, normalize=True):
    sensors = PAIR_SENSORS[pair]

    pair_scores = {}
    for sensor in sensors:
        sensor_dir = os.path.join(root, pair, sensor, obj)
        frames, scores = frame_score_sequence(sensor_dir, episode, baseline_n=baseline_n)

        if frames is None:
            print(f"[skip] missing {pair}/{sensor}/{obj}/ep{episode:04d}")
            return False

        if normalize:
            scores_plot = minmax_norm(scores)
        else:
            scores_plot = scores

        pair_scores[sensor] = (frames, scores_plot, scores)

    # combine only inside this pair
    combined_by_frame = {}
    for sensor, (frames, scores_plot, raw_scores) in pair_scores.items():
        for fr, sc in zip(frames, scores_plot):
            combined_by_frame.setdefault(fr, []).append(sc)

    common_frames = np.array(sorted(combined_by_frame.keys()))
    combined = np.array([np.mean(combined_by_frame[fr]) for fr in common_frames])

    plt.figure(figsize=(12, 5))

    for sensor, (frames, scores_plot, raw_scores) in pair_scores.items():
        plt.plot(frames, scores_plot, linewidth=1.8, alpha=0.8, label=f"{sensor}")

    plt.plot(common_frames, combined, color="black", linewidth=3, label="pair mean")

    plt.xlabel("frame")
    ylabel = "normalized contact score" if normalize else "raw contact score"
    plt.ylabel(ylabel)
    plt.title(f"{pair} | {obj} | ep{episode:04d}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pair}_{obj}_ep{episode:04d}.png")
    plt.savefig(out_path, dpi=180)
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Path to latent_data_new_processed")
    parser.add_argument("--object", default="small_ball")
    parser.add_argument("--out_dir", default="contact_score_plots")
    parser.add_argument("--baseline_n", type=int, default=1,
                        help="Number of first frames used as baseline")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Plot raw scores instead of per-episode min-max normalized scores")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Optional limit, e.g. 150")
    args = parser.parse_args()

    normalize = not args.no_normalize

    for pair, sensors in PAIR_SENSORS.items():
        print(f"\n=== Processing {pair} ===")

        # episode list: use first sensor folder
        first_sensor_dir = os.path.join(args.root, pair, sensors[0], args.object)
        if not os.path.isdir(first_sensor_dir):
            print(f"[skip] missing folder: {first_sensor_dir}")
            continue

        episodes = get_available_episodes(first_sensor_dir)

        if args.max_episodes is not None:
            episodes = episodes[:args.max_episodes]

        pair_out_dir = os.path.join(args.out_dir, pair, args.object)
        os.makedirs(pair_out_dir, exist_ok=True)

        count = 0
        for ep in episodes:
            ok = plot_one_episode(
                root=args.root,
                pair=pair,
                obj=args.object,
                episode=ep,
                out_dir=pair_out_dir,
                baseline_n=args.baseline_n,
                normalize=normalize,
            )
            if ok:
                count += 1

        print(f"saved {count} plots under: {pair_out_dir}")


if __name__ == "__main__":
    main()