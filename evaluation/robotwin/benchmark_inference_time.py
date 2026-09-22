"""Measure Zero-WAM inference time without the Robotwin simulator.

Replays the request sequence of eval_policy_client_openpi.py against a running
inference server (launch_server.sh) with synthetic camera frames. Compute cost
depends on tensor shapes and the ICL context length, not on pixel content, so
the timings match a real rollout.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluation.robotwin.robotwin_icl_human_videos import ROBOTWIN_ICL_HUMAN_VIDEOS
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

CAM_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def resolve_icl_latent_path(video_path, latent_root):
    # Same mirrored layout as eval_policy_client_openpi.resolve_icl_latent_path,
    # which cannot be imported without the Robotwin simulator.
    parts = Path(video_path).parts
    relative_parts = parts[parts.index("human_data") + 1 :]
    if relative_parts[0] == Path(latent_root).name:
        relative_parts = relative_parts[1:]
    candidate = Path(latent_root).joinpath(*relative_parts).with_suffix(".pth")
    return str(candidate) if candidate.exists() else ""


def fake_obs(rng, height, width, prompt):
    obs = {k: rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for k in CAM_KEYS}
    obs["observation.state"] = np.zeros(14, dtype=np.float64)
    obs["task"] = prompt
    return obs


def timed_infer(model, request):
    start = time.perf_counter()
    ret = model.infer(request)
    total_ms = (time.perf_counter() - start) * 1000
    return ret, ret["server_timing"]["infer_ms"], total_ms


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--task", type=str, default="place_object_scale")
    parser.add_argument("--icl_latent_root", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--icl_guidance_scale", type=float, default=5.0)
    parser.add_argument("--video_guidance_scale", type=float, default=-1.0)
    parser.add_argument("--cam_height", type=int, default=240)
    parser.add_argument("--cam_width", type=int, default=320)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    prompt = args.task.replace("_", " ")
    icl_video_path = ROBOTWIN_ICL_HUMAN_VIDEOS[args.task][0]
    icl_latent_path = resolve_icl_latent_path(icl_video_path, args.icl_latent_root)
    if not icl_latent_path:
        raise FileNotFoundError(f"No precomputed ICL latent for {icl_video_path}")
    print(f"[bench] task={args.task} latent={icl_latent_path}", flush=True)

    model = WebsocketClientPolicy(port=args.port)
    records = []
    for episode in range(args.episodes):
        _, server_ms, total_ms = timed_infer(
            model,
            dict(
                reset=True,
                prompt=prompt,
                save_visualization=False,
                use_icl=True,
                icl_video_path=icl_video_path,
                icl_latent_path=icl_latent_path,
                video_guidance_scale=args.video_guidance_scale,
                icl_guidance_scale=args.icl_guidance_scale,
            ),
        )
        records.append(dict(episode=episode, chunk=-1, call="reset", server_ms=server_ms, total_ms=total_ms))
        print(f"[bench] ep{episode} reset          server={server_ms:9.1f} ms  roundtrip={total_ms:9.1f} ms", flush=True)

        first_obs = fake_obs(rng, args.cam_height, args.cam_width, prompt)
        for chunk in range(args.chunks):
            ret, server_ms, total_ms = timed_infer(
                model,
                dict(
                    obs=first_obs,
                    prompt=prompt,
                    save_visualization=False,
                    video_guidance_scale=args.video_guidance_scale,
                    action_guidance_scale=1,
                ),
            )
            action = ret["action"]
            records.append(dict(episode=episode, chunk=chunk, call="infer", server_ms=server_ms, total_ms=total_ms))
            infer_ms = server_ms

            # The client returns one key frame every action_per_frame // 4 steps
            # and skips the first (observed) frame of the first chunk.
            key_frames_per_latent = 4
            latent_frames = action.shape[1] - (1 if chunk == 0 else 0)
            key_frame_list = [
                fake_obs(rng, args.cam_height, args.cam_width, prompt)
                for _ in range(latent_frames * key_frames_per_latent)
            ]
            _, server_ms, total_ms = timed_infer(
                model,
                dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, save_visualization=False, state=action),
            )
            records.append(dict(episode=episode, chunk=chunk, call="kv_cache", server_ms=server_ms, total_ms=total_ms))
            print(
                f"[bench] ep{episode} chunk{chunk:03d} action={tuple(action.shape)} "
                f"infer={infer_ms:9.1f} ms  kv_cache={server_ms:8.1f} ms",
                flush=True,
            )

    # The first chunk of the first episode pays torch.compile / autotune costs.
    summary = {}
    for call in ("reset", "infer", "kv_cache"):
        warm = [
            r["server_ms"]
            for r in records
            if r["call"] == call and not (r["episode"] == 0 and r["chunk"] <= 0 and call != "reset")
        ]
        if warm:
            summary[call] = summarize(warm)
    infer_mean = summary["infer"]["mean_ms"]
    kv_mean = summary["kv_cache"]["mean_ms"]
    actions_per_chunk = int(action.shape[1] * action.shape[2])
    summary["per_chunk_cycle_ms"] = infer_mean + kv_mean
    summary["actions_per_chunk"] = actions_per_chunk
    summary["effective_action_hz"] = actions_per_chunk / ((infer_mean + kv_mean) / 1000)

    print("\n[bench] ===== summary (server-side ms, warm-up chunk excluded) =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(args=vars(args), summary=summary, records=records), indent=2))
        print(f"[bench] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
