"""
CausalRec-Bench — Download All Files

Downloads all dataset files and pre-trained
models from Google Drive automatically.

Usage:
    pip install gdown
    python download_data.py

Downloads approximately 1.7 GB total.
"""

import os
import sys
import subprocess

def install_gdown():
    subprocess.run([
        sys.executable, '-m', 'pip',
        'install', 'gdown', '--quiet'
    ])

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    install_gdown()
    import gdown

print("=" * 60)
print("CausalRec-Bench — Download Script")
print("=" * 60)
print()

# ─── ALL FILE IDs FROM GOOGLE DRIVE ───────

ALL_FILES = {
    # ── Data files ──
    'data/users.csv':
        '1ukl5MxpvJMzxdVzonSygYNJXZUlJKEwm',
    'data/items.csv':
        '1zOTil0pEACgFY8a75q4zVbV03sJZJyMO',
    'data/items_ecommerce.csv':
        '1Eok6g1S4IbW2ZliqnXMjh9mUUoRH5OHt',
    'data/items_streaming.csv':
        '1IhpyRFDH59IZhFIDj01wcUDKOpZsY83O',
    'data/interactions.csv':
        '1s7TPdB8v4q0MLpfxCRJJo6zx4Fa9pd7z',
    'data/train.csv':
        '1YdlbY2_Bc9G7yWwXNzi6OWipsRWDbFXN',
    'data/val.csv':
        '1hTtBsPLEpZd8Daow8knOZ4VNRemsJu_6',
    'data/test.csv':
        '1-6W_uZdO-Mhp7bwoyFBppd6bHqlSbDey',
    'data/cold_start.csv':
        '1t1cMdZvmGYnJw3qodlcjCHuP0pyk6HAH',
    'data/cold_start_profiles.csv':
        '1yep1pY92aluWj0Alv7TJOvP10wtOEyhu',
    'data/level1_simple.csv':
        '1oWmpUvKmbVvEwygVDm8WCJoZeMsq72qY',
    'data/level2_medium.csv':
        '1d3vaMVzuzAsMgYojIC5M4jBR9TCTCsY5',
    'data/level3_hard.csv':
        '16NI5f91WS-_HN8d4CriwUH-HSvIZdAqc',
    'data/winter_cold.csv':
        '1_MCr2SLsHWAzswLJtY9ZOPu6HyT16XiE',
    'data/summer_cold.csv':
        '1C1HScMMwDD8SxlUv11bgo2GCTHFRv2WV',
    'data/autumn_cold.csv':
        '1Y8tRIvczbCg7tkpZHtZqVOSSPZgjeDWV',
    'data/spring_cold.csv':
        '1jqb7pg3h1p-OLdFEqLs9EMJeMhIs4EJH',
    'data/ecom_cold.csv':
        '1PS2SIU86qdLojU305tfLK0RjeUbEL5eo',
    'data/stream_cold.csv':
        '1qvT4rZLlMUTPksJguAg4URlzYUveTik1',
    'data/ecom_only.csv':
        '1gFWBV75tbzKI2pxpWcxKpmnneJVcpLIp',
    'data/stream_only.csv':
        '17MXrCVvgV3yRTNvqaJS-QeRHXOUA3EGO',
    'data/high_position.csv':
        '1J75jESNq_Kl6vEtwiRZmG3-p0vk22zyM',
    'data/low_position.csv':
        '1vlZ6mveiNfYn06H68kXqM9gNUWEyJsNO',
    'data/validation_report.csv':
        '1LZoCXOH8HPFfEG_V6ZjWCHbAXSADQh_n',

    # ── Pre-trained models ──
    'pretrained_models/fmf_std_U.npy':
        '1q8-lY12wg63mwf1KADC89RMJFKh6_GG3',
    'pretrained_models/fmf_std_V.npy':
        '174K5UwAjPTfDKEewdPkhQ8djg20PzsvH',
    'pretrained_models/fmf_caus_U.npy':
        '1t8SRut5tehIkQv5H288iEKGExbo1as2B',
    'pretrained_models/fmf_caus_V.npy':
        '1DV8jwxa0PKNcvI4BWwmmTJtSzox3Qne4',
    'pretrained_models/lgcn_std.pt':
        '1oW6DKhuqfe53m4H9Q6PwFDugy1JRSyDY',
    'pretrained_models/lgcn_caus.pt':
        '1wCzpUsYF9QbO8TD_F6JhQgCJcpYlW5lY',

    # ── Figures ──
    'figures/validation_charts_v2.png':
        '1kfMsv19njZJl2VuX-KylGoTZ8qu5UEV9',
    'figures/final_results_v2.png':
        '1c50e0EGZI6mnZffUPH9VV2KSoU87gdKA',
}

# ─── CREATE FOLDERS ───────────────────────
for folder in ['data', 'pretrained_models',
               'figures', 'results']:
    os.makedirs(folder, exist_ok=True)

# ─── CHECK WHAT IS ALREADY DOWNLOADED ─────
already = []
to_download = {}

for path, fid in ALL_FILES.items():
    if os.path.exists(path):
        size = os.path.getsize(path)/1024/1024
        if size > 0.01:
            already.append((path, size))
        else:
            to_download[path] = fid
    else:
        to_download[path] = fid

if already:
    print("Already downloaded:")
    for path, size in already:
        print(f"  {path} ({size:.0f} MB)")
    print()

if not to_download:
    print("All files already present.")
    print("Run: python VERIFY_RESULTS.py")
    sys.exit(0)

# ─── SHOW DOWNLOAD PLAN ───────────────────
total_count = len(to_download)
print(f"Files to download: {total_count}")
print()

# Calculate approximate total size
size_map = {
    'interactions.csv': 280,
    'train.csv': 197,
    'level3_hard.csv': 280,
    'level2_medium.csv': 197,
    'ecom_only.csv': 147,
    'stream_only.csv': 133,
    'high_position.csv': 87,
    'low_position.csv': 81,
    'cold_start.csv': 54,
    'test.csv': 56,
    'level1_simple.csv': 58,
    'val.csv': 28,
    'ecom_cold.csv': 28,
    'stream_cold.csv': 26,
}
total_mb = sum(
    size_map.get(
        os.path.basename(p), 5
    )
    for p in to_download
)
print(
    f"Estimated download size: "
    f"~{total_mb:,} MB "
    f"({total_mb/1024:.1f} GB)"
)
print()

response = input(
    "Start download? (yes/no): "
).strip().lower()

if response not in ['yes', 'y']:
    print("Download cancelled.")
    sys.exit(0)

print()
print("Downloading files...")
print()

# ─── DOWNLOAD FILES ───────────────────────
success = []
failed = []

# Group by category for display
categories = {
    'Core data files': [
        'data/users.csv',
        'data/items.csv',
        'data/items_ecommerce.csv',
        'data/items_streaming.csv',
        'data/validation_report.csv',
        'data/cold_start_profiles.csv',
    ],
    'Interaction files': [
        'data/interactions.csv',
        'data/train.csv',
        'data/val.csv',
        'data/test.csv',
        'data/cold_start.csv',
    ],
    'Difficulty levels': [
        'data/level1_simple.csv',
        'data/level2_medium.csv',
        'data/level3_hard.csv',
    ],
    'Seasonal splits': [
        'data/winter_cold.csv',
        'data/summer_cold.csv',
        'data/autumn_cold.csv',
        'data/spring_cold.csv',
    ],
    'Domain splits': [
        'data/ecom_cold.csv',
        'data/stream_cold.csv',
        'data/ecom_only.csv',
        'data/stream_only.csv',
    ],
    'Position splits': [
        'data/high_position.csv',
        'data/low_position.csv',
    ],
    'Pre-trained models': [
        'pretrained_models/fmf_std_U.npy',
        'pretrained_models/fmf_std_V.npy',
        'pretrained_models/fmf_caus_U.npy',
        'pretrained_models/fmf_caus_V.npy',
        'pretrained_models/lgcn_std.pt',
        'pretrained_models/lgcn_caus.pt',
    ],
    'Figures': [
        'figures/validation_charts_v2.png',
        'figures/final_results_v2.png',
    ],
}

for category, files in categories.items():
    print(f"{category}:")
    for path in files:
        if path not in to_download:
            size = os.path.getsize(path)/1024/1024
            print(
                f"  SKIP {os.path.basename(path)}"
                f" (exists, {size:.0f} MB)"
            )
            continue

        fid = to_download[path]
        url = (
            f'https://drive.google.com/'
            f'uc?id={fid}'
        )
        fname = os.path.basename(path)

        try:
            print(
                f"  Downloading {fname}...",
                end='', flush=True
            )
            gdown.download(
                url, path,
                quiet=True
            )
            if os.path.exists(path):
                size = (
                    os.path.getsize(path)
                    / 1024 / 1024
                )
                print(f" {size:.0f} MB OK")
                success.append(path)
            else:
                print(" FAILED")
                failed.append(path)
        except Exception as e:
            print(f" ERROR: {e}")
            failed.append(path)

    print()

# ─── SUMMARY ──────────────────────────────
print("=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(
    f"Successfully downloaded: "
    f"{len(success)} files"
)

if failed:
    print(
        f"Failed: {len(failed)} files"
    )
    for f in failed:
        print(f"  {f}")
    print()
    print(
        "For failed files download manually:"
    )
    print(
        "https://drive.google.com/drive/"
        "folders/"
        "193xUcjZHh03Hal0v1F-z7lhpKQ0yCKNh"
    )
else:
    print()
    print("All files downloaded successfully")
    print()
    print("Next step:")
    print(
        "  python VERIFY_RESULTS.py"
    )
    print()
    print(
        "This will load our pre-trained "
        "models and verify all paper "
        "results are real."
    )
