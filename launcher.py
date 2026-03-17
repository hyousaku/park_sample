#!/usr/bin/env python3
"""
モード選択ランチャー
起動時にどのモードで実行するかを選択できます
"""

import sys
import os

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
    while True:
        show_menu()
        
        try:
            choice = input("選択 (0-2): ").strip()
            
            if choice == "0":
                print("\n終了します。")
                sys.exit(0)
            
            elif choice == "1":
                print("\nグリッドモードを起動します...")
                os.system("python main.py")
                break
            
            elif choice == "2":
                print("\nバブルモードを起動します...")
                os.system("python bubble_mode.py")
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
