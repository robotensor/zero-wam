# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from pathlib import Path

from easydict import EasyDict

from .shared_config import va_shared_cfg


def load_robotwin_norm_stat(action_stats_path):
    action_stats_path = Path(action_stats_path)
    with action_stats_path.open(encoding='utf-8') as stats_file:
        action_stats = json.load(stats_file)
    if action_stats.get('method') != 'abs':
        raise ValueError(
            f'Robotwin action stats must use method=abs: {action_stats_path}'
        )
    hand_stats = action_stats['norm_stats']['action.hand.position']
    effector_stats = action_stats['norm_stats']['action.effector.position']
    return {
        'q01': hand_stats['q01'] + [0.0] * 14 + effector_stats['q01'],
        'q99': hand_stats['q99'] + [0.0] * 14 + effector_stats['q99'],
    }


va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.model_path = os.environ.get(
    "MODEL_PATH",
    "/path/to/zero-wam-posttrain-robotwin",
)
_default_empty_text_emb_path = (
    Path(__file__).resolve().parents[1] / 'assets' / 'empty_text_emb.pt'
)
va_robotwin_cfg.empty_text_emb_path = os.environ.get(
    "EMPTY_TEXT_EMB_PATH", str(_default_empty_text_emb_path)
)
va_robotwin_cfg.use_icl_model = True

va_robotwin_cfg.attn_window = 64
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'franka'

# The released Robotwin checkpoint was trained with the original Franka view
# layout: three equally sized camera latents concatenated along width.
va_robotwin_cfg.height = 224
va_robotwin_cfg.width = 288
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = float(os.environ.get("TARGET_TEXT_CFG", "-1"))
va_robotwin_cfg.action_guidance_scale = 1
va_robotwin_cfg.icl_guidance_scale = float(os.environ.get("ICL_CFG", "5"))
va_robotwin_cfg.icl_rope_h = 24
va_robotwin_cfg.icl_height = 320
va_robotwin_cfg.icl_width = 480
va_robotwin_cfg.icl_fps = 12

va_robotwin_cfg.num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = int(
    os.environ.get("ACTION_NUM_INFERENCE_STEPS", "50")
)

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
_action_stats_path = (
    Path(__file__).resolve().parents[1]
    / 'assets'
    / 'norm_stats'
    / 'robotwin_icl.json'
)
va_robotwin_cfg.action_stats_path = str(_action_stats_path)
va_robotwin_cfg.norm_stat = load_robotwin_norm_stat(_action_stats_path)
