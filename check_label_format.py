#!/usr/bin/env python3
"""检查 dataset 中 test/train/valid 三个文件夹的 labels 是否都是长方形格式（YOLO HBB: 5 个值 / OBB: 9 个值）。"""

import os
from collections import Counter
from pathlib import Path

DATASET_DIR = Path("dataset")
SPLITS = ["train", "valid", "test"]
# YOLO 标签格式:
#   HBB (水平矩形框): class x_center y_center width height      -> 5 个值
#   OBB (旋转矩形框): class x1 y1 x2 y2 x3 y3 x4 y4            -> 9 个值
EXPECTED_COUNTS = {5, 9}


def is_valid_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_split(split: str):
    labels_dir = DATASET_DIR / split / "labels"
    if not labels_dir.is_dir():
        return None

    files = sorted(p for p in labels_dir.iterdir() if p.suffix == ".txt")
    total_files = len(files)
    total_lines = 0
    empty_files = []
    bad_files = []
    field_count_counter = Counter()
    non_numeric_lines = []
    class_id_counter = Counter()
    out_of_range = []
    not_in_expected_shape = []

    for txt in files:
        try:
            content = txt.read_text(encoding="utf-8").strip()
        except Exception as exc:
            bad_files.append((txt.name, f"读取失败: {exc}"))
            continue

        if not content:
            empty_files.append(txt.name)
            continue

        lines = content.splitlines()
        for line_no, line in enumerate(lines, start=1):
            total_lines += 1
            parts = line.split()
            field_count_counter[len(parts)] += 1

            if len(parts) not in EXPECTED_COUNTS:
                not_in_expected_shape.append((txt.name, line_no, len(parts), line))
                continue

            if not all(is_valid_number(p) for p in parts):
                non_numeric_lines.append((txt.name, line_no, line))
                continue

            cls = int(float(parts[0]))
            class_id_counter[cls] += 1
            # 6 个类别 -> class id 0~5
            if cls < 0 or cls > 5:
                out_of_range.append((txt.name, line_no, cls, line))

            # 对 HBB(5) 检查坐标范围; OBB(9) 也做同样检查
            coords = [float(p) for p in parts[1:]]
            for v in coords:
                if v < 0.0 or v > 1.0:
                    out_of_range.append((txt.name, line_no, v, f"坐标越界: {line}"))
                    break

    print(f"\n========== {split.upper()} ==========")
    print(f"标签文件总数        : {total_files}")
    print(f"总标注行数          : {total_lines}")
    print(f"空标签文件数        : {len(empty_files)}")
    print(f"无法读取的文件数    : {len(bad_files)}")
    print(f"每行字段值分布      : {dict(field_count_counter)}")
    print(f"非数字内容行数      : {len(non_numeric_lines)}")
    print(f"字段数非 5/9 的行数 : {len(not_in_expected_shape)}")
    print(f"class 越界 (0~5)    : {len(out_of_range)}")
    print(f"各类别出现次数      : {dict(sorted(class_id_counter.items()))}")

    if empty_files:
        print(f"  空文件样例(前 5): {empty_files[:5]}")
    if bad_files:
        print(f"  无法读取样例(前 5): {bad_files[:5]}")
    if not_in_expected_shape:
        print(f"  字段数异常样例(前 5): {not_in_expected_shape[:5]}")
    if non_numeric_lines:
        print(f"  非数字行样例(前 5): {non_numeric_lines[:5]}")
    if out_of_range:
        print(f"  class 越界样例(前 5): {out_of_range[:5]}")

    # 结论
    if (len(not_in_expected_shape) == 0 and len(non_numeric_lines) == 0
            and len(out_of_range) == 0 and len(bad_files) == 0):
        shape = "HBB(5 字段)" if field_count_counter and max(field_count_counter) == 5 and 9 not in field_count_counter \
            else "OBB(9 字段)" if field_count_counter and 5 not in field_count_counter and 9 in field_count_counter \
            else "混合/其他"
        print(f"  ✅ 所有标签都是合法的长方形格式({shape})")
    else:
        print(f"  ❌ 存在异常标签")


def main():
    for split in SPLITS:
        check_split(split)


if __name__ == "__main__":
    main()
