from PIL import Image, ImageDraw
import numpy as np


def _xy_to_px(x, y, W, H):
    u = int((x + 1.0) * 0.5 * (W - 1))
    v = int((1.0 - (y + 1.0) * 0.5) * (H - 1))
    return u, v


def render_frame(env, width=640, height=480, score_left=0, score_right=0):
    img = Image.new("RGB", (width, height), (18, 20, 24))
    d = ImageDraw.Draw(img)
    for y in range(0, height, 24):
        d.rectangle([width // 2 - 2, y, width // 2 + 2, y + 12], fill=(70, 70, 80))
    d.rectangle([0, 0, 6, height], outline=(60, 60, 70))
    d.rectangle([width - 7, 0, width - 1, height], outline=(60, 60, 70))
    scale_y = (height - 1) / 2.0
    pad_h_px = int(env.paddle_h * scale_y)
    pad_w_px = 10
    xL = 12
    xR = width - 12
    _, vL = _xy_to_px(-1.0, env.yl, width, height)
    d.rectangle(
        [
            xL - pad_w_px // 2,
            vL - pad_h_px // 2,
            xL + pad_w_px // 2,
            vL + pad_h_px // 2,
        ],
        fill=(180, 220, 250),
    )
    _, vR = _xy_to_px(+1.0, env.yr, width, height)
    d.rectangle(
        [
            xR - pad_w_px // 2,
            vR - pad_h_px // 2,
            xR + pad_w_px // 2,
            vR + pad_h_px // 2,
        ],
        fill=(255, 190, 120),
    )
    uB, vB = _xy_to_px(env.ball_x, env.ball_y, width, height)
    r = 6
    d.ellipse([uB - r, vB - r, uB + r, vB + r], fill=(240, 240, 240))
    s = f"{score_left} : {score_right}"
    d.rectangle([width // 2 - 50, 8, width // 2 + 50, 34], fill=(30, 32, 38))
    d.text((width // 2 - 20, 12), s, fill=(230, 230, 240))
    return np.array(img)
