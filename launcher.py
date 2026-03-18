#!/usr/bin/env python3
"""
モード選択ランチャー
起動時にどのモードで実行するかを選択できます
"""

import sys
import os
import cv2


def detect_cameras():
    """利用可能なカメラを検出して一覧を返す"""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((i, w, h))
            cap.release()
    return cameras


def select_camera():
    """カメラ選択メニューを表示して選択されたインデックスを返す"""
    print("\n" + "="*50)
    print("  カメラを選択してください")
    print("="*50)
    print("\nカメラを検出しています...")
    cameras = detect_cameras()

    if not cameras:
        print("  カメラが見つかりません。デフォルト(index 0)を使用します。")
        return 0

    print()
    for idx, (cam_index, w, h) in enumerate(cameras, 1):
        print(f"  {idx}. カメラ {cam_index}  ({w}x{h})")
    print()
    print("="*50)

    while True:
        try:
            choice = input(f"選択 (1-{len(cameras)}): ").strip()
            n = int(choice)
            if 1 <= n <= len(cameras):
                selected = cameras[n - 1][0]
                print(f"\nカメラ {selected} を使用します。")
                return selected
            else:
                print(f"1〜{len(cameras)}の数字を入力してください。")
        except ValueError:
            print("数字を入力してください。")
        except KeyboardInterrupt:
            print("\n\n終了します。")
            sys.exit(0)


def show_menu():
    """モード選択メニューを表示"""
    print("\n" + "="*50)
    print("  人検出インタラクティブアプリケーション")
    print("="*50)
    print("\nモードを選択してください：")
    print()
    print("  1. グリッドモード（main.py）")
    print("     - 16x9グリッドに頭部が入るとピアノ音が鳴る")
    print("     - 12秒ごとに音階がランダムに変更")
    print()
    print("  2. バブルモード（bubble_mode.py）")
    print("     - ふわふわ移動する円に頭を当てると打楽器音が鳴る")
    print("     - 常時5個の円が画面上を移動")
    print()
    print("  0. 終了")
    print()
    print("="*50)


def main():
    """メイン処理"""
    camera_index = select_camera()
    os.environ["CAMERA_INDEX"] = str(camera_index)

    while True:
        show_menu()

        try:
            choice = input("選択 (0-2): ").strip()

            if choice == "0":
                print("\n終了します。")
                sys.exit(0)

            elif choice == "1":
                print("\nグリッドモードを起動します...")
                os.system(f'"{sys.executable}" main.py')
                break

            elif choice == "2":
                print("\nバブルモードを起動します...")
                os.system(f'"{sys.executable}" bubble_mode.py')
                break

            else:
                print("\n無効な選択です。0-2の数字を入力してください。")
                input("Enterキーを押して続行...")

        except KeyboardInterrupt:
            print("\n\n終了します。")
            sys.exit(0)
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            input("Enterキーを押して続行...")

if __name__ == "__main__":
    main()
