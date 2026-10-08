"""Generates a high-fidelity 1200x800 demo.gif for nn-htop."""

import math
import os
import sys
import time
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nn_htop.harness import ThreeLayerMLP
from nn_htop.monitor import Monitor


def create_demo_gif(output_path: str = "demo.gif", total_seconds: float = 10.0, fps: int = 5):
    width, height = 1200, 800
    bg_color = (17, 20, 24)       # #111418
    banner_bg = (24, 30, 38)     # #181e26
    border_color = (44, 56, 72)  # #2c3848
    text_color = (224, 230, 237) # #e0e6ed
    cyan = (0, 210, 255)
    green = (0, 220, 130)
    yellow = (240, 180, 40)
    magenta = (220, 80, 220)
    dim_gray = (100, 115, 130)
    row_alt = (22, 27, 34)

    # Load font
    try:
        # Standard Windows monospace fonts
        font_large = ImageFont.truetype("consola.ttf", 22)
        font_bold = ImageFont.truetype("consolab.ttf", 16)
        font_main = ImageFont.truetype("consola.ttf", 15)
        font_small = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        font_large = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_small = ImageFont.load_default()

    model = ThreeLayerMLP()
    monitor = Monitor(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_frames = int(total_seconds * fps)
    frames = []

    print(f"Rendering {total_frames} frames ({total_seconds}s at {fps} FPS)...")

    for frame_idx in range(total_frames):
        # Perform 2 training steps per frame
        for _ in range(2):
            inputs = torch.randn(32, 16)
            targets = torch.randint(0, 4, (32,))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            monitor.record_step(loss=loss.item())

        stats = monitor.get_stats()
        layer_names = monitor.get_layer_names()

        # Create Image Canvas
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # 1. Top Window Bar
        draw.rectangle([(0, 0), (width, 36)], fill=(27, 33, 41))
        # Window buttons (macOS/modern style)
        draw.ellipse([(14, 12), (26, 24)], fill=(255, 95, 86))
        draw.ellipse([(34, 12), (46, 24)], fill=(255, 189, 46))
        draw.ellipse([(54, 12), (66, 24)], fill=(39, 201, 63))
        # Title
        draw.text((width // 2 - 140, 8), "nn-htop: Neural Network Telemetry", fill=text_color, font=font_bold)

        # 2. Stats Banner Panel
        banner_top = 46
        banner_bot = 96
        draw.rectangle([(16, banner_top), (width - 16, banner_bot)], fill=banner_bg, outline=border_color, width=1)

        step = monitor.step_count
        loss_val = monitor.latest_loss
        draw.text((28, banner_top + 14), "nn-htop", fill=cyan, font=font_bold)
        draw.text((105, banner_top + 14), "│", fill=border_color, font=font_main)
        draw.text((120, banner_top + 14), "Status: ", fill=dim_gray, font=font_main)
        draw.text((178, banner_top + 14), "● RUNNING", fill=green, font=font_bold)
        draw.text((270, banner_top + 14), "│", fill=border_color, font=font_main)
        draw.text((285, banner_top + 14), f"Step: ", fill=dim_gray, font=font_main)
        draw.text((328, banner_top + 14), f"{step:5d}", fill=(255, 255, 255), font=font_bold)
        draw.text((395, banner_top + 14), "│", fill=border_color, font=font_main)
        draw.text((410, banner_top + 14), "Loss: ", fill=dim_gray, font=font_main)
        draw.text((455, banner_top + 14), f"{loss_val:.4f}", fill=magenta, font=font_bold)
        draw.text((535, banner_top + 14), "│", fill=border_color, font=font_main)
        draw.text((550, banner_top + 14), f"Layers: {len(layer_names)}/{len(layer_names)} learning", fill=green, font=font_main)
        draw.text((width - 180, banner_top + 14), f"FPS: {fps}.0", fill=cyan, font=font_main)

        # 3. Main DataTable
        table_top = 110
        table_bot = height - 45
        draw.rectangle([(16, table_top), (width - 16, table_bot)], fill=bg_color, outline=border_color, width=1)

        # Columns Configuration
        cols = [
            ("Layer Name", 35, 180),
            ("Type", 220, 160),
            ("Mean Activation", 390, 190),
            ("Gradient Norm", 590, 180),
            ("Health", 780, 160),
            ("Grad Activity", 950, 220),
        ]

        # Table Header Row
        header_y = table_top + 8
        draw.rectangle([(18, header_y - 4), (width - 18, header_y + 26)], fill=(27, 33, 41))
        for col_name, col_x, _ in cols:
            draw.text((col_x, header_y), col_name, fill=cyan, font=font_bold)
        draw.line([(18, header_y + 26), (width - 18, header_y + 26)], fill=border_color, width=1)

        # Layer Rows
        row_y = header_y + 36
        row_height = 42

        for i, name in enumerate(layer_names):
            s = stats.get(name)
            if not s:
                continue

            # Row zebra striping
            if i % 2 == 1:
                draw.rectangle([(18, row_y - 4), (width - 18, row_y + row_height - 6)], fill=row_alt)

            # Layer Name
            draw.text((cols[0][1], row_y + 6), name, fill=cyan, font=font_bold)
            # Type
            draw.text((cols[1][1], row_y + 6), s.layer_type, fill=text_color, font=font_main)
            # Mean Activation
            act_text = f"{s.mean_activation:+.5f}"
            act_color = green if abs(s.mean_activation) > 0.05 else dim_gray
            draw.text((cols[2][1], row_y + 6), act_text, fill=act_color, font=font_main)
            # Gradient Norm
            grad_text = f"{s.grad_norm:.5f}"
            grad_color = (255, 255, 255) if s.grad_norm > 0.01 else yellow
            draw.text((cols[3][1], row_y + 6), grad_text, fill=grad_color, font=font_main)

            # Health Badge
            health_color = green
            health_text = "LEARNING"
            if s.grad_norm < 1e-4:
                health_color = yellow
                health_text = "VANISHING"
            elif s.grad_norm > 50.0:
                health_color = (255, 80, 80)
                health_text = "EXPLODING"
            elif s.layer_type == "ReLU" and abs(s.mean_activation) < 1e-6:
                health_color = yellow
                health_text = "DEAD RELU"

            # Badge pill
            badge_x = cols[4][1]
            draw.rectangle([(badge_x, row_y + 4), (badge_x + 95, row_y + 24)], outline=health_color, width=1)
            draw.text((badge_x + 8, row_y + 6), health_text, fill=health_color, font=font_small)

            # Grad Activity Bar (ASCII Meter)
            meter_x = cols[5][1]
            ratio = min(max(s.grad_norm / 0.25, 0.0), 1.0)
            bar_len = 12
            filled = int(round(ratio * bar_len))
            meter_chars = "■" * filled + "·" * (bar_len - filled)
            bar_color = green if ratio > 0.3 else cyan
            draw.text((meter_x, row_y + 6), f"[{meter_chars}]", fill=bar_color, font=font_main)

            row_y += row_height

        # 4. Footer Bar
        footer_top = height - 38
        draw.rectangle([(0, footer_top), (width, height)], fill=(27, 33, 41))
        footer_keys = [
            ("q", "Quit"),
            ("space", "Pause/Resume"),
            ("r", "Reset Step"),
        ]
        curr_x = 20
        for k, action in footer_keys:
            draw.rectangle([(curr_x, footer_top + 6), (curr_x + 24 + len(k)*6, footer_top + 28)], fill=(44, 56, 72))
            draw.text((curr_x + 6, footer_top + 8), k, fill=(255, 255, 255), font=font_bold)
            curr_x += 32 + len(k)*6
            draw.text((curr_x, footer_top + 8), action, fill=dim_gray, font=font_main)
            curr_x += len(action) * 9 + 25

        frames.append(img)

    monitor.remove_hooks()

    # Save as animated GIF
    print(f"Saving animated GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully generated {output_path} ({file_size_mb:.2f} MB, {total_frames} frames)!")


if __name__ == "__main__":
    create_demo_gif("demo.gif", total_seconds=10.0, fps=5)
