"""
build.py — 跨平台 C 扩展编译脚本

自动检测平台，调用正确的编译器编译 _features.c，并验证结果。

用法:
    python build.py

支持平台:
    Windows  — 需要 MSYS2 + MinGW-w64 gcc（默认路径 C:\\msys64\\mingw64\\bin\\gcc.exe）
               如安装到其他路径，设置环境变量 MINGW_BIN=C:\\path\\to\\mingw64\\bin
    Linux    — 需要 gcc + python3-dev（apt install build-essential python3-dev）
    macOS    — 需要 Xcode Command Line Tools（xcode-select --install）
"""

import importlib.util
import os
import platform
import subprocess
import sys


MINGW_DEFAULT = r"C:\msys64\mingw64\bin"


def _find_mingw_bin() -> str | None:
    """返回 MinGW64 bin 目录路径，找不到返回 None。"""
    # 优先使用环境变量
    env_path = os.environ.get("MINGW_BIN", "")
    if env_path and os.path.isfile(os.path.join(env_path, "gcc.exe")):
        return env_path

    # 检查默认路径
    if os.path.isfile(os.path.join(MINGW_DEFAULT, "gcc.exe")):
        return MINGW_DEFAULT

    # 尝试从 PATH 里找
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isfile(os.path.join(p, "gcc.exe")):
            return p

    return None


def _verify() -> bool:
    """验证 _features 模块可以正常 import 并运行。"""
    # 强制重新查找（编译前可能已加载失败的状态）
    if "_features" in sys.modules:
        del sys.modules["_features"]

    spec = importlib.util.find_spec("_features")
    if spec is None:
        print("[build] 警告: 编译成功但找不到 _features 模块（可能需要重启 Python）")
        return False

    import _features
    result = _features.extract_features("hello 你好 123")
    cjk, letter, digit, punct, space, word = result
    print(f"[build] 验证通过: extract_features('hello 你好 123')")
    print(f"[build]   → cjk={cjk}, letter={letter}, digit={digit}, "
          f"punct={punct}, space={space}, word={word}")
    return True


def main():
    plat = platform.system()
    print(f"[build] 平台: {plat} ({platform.machine()})")
    print(f"[build] Python: {sys.executable}  ({sys.version.split()[0]})")

    env = None
    if plat == "Windows":
        mingw_bin = _find_mingw_bin()
        if mingw_bin is None:
            print("\n[build] 错误: 未找到 MSYS2 MinGW-w64 gcc。")
            print("  请按以下步骤安装：")
            print("    1. 下载 MSYS2: https://www.msys2.org/")
            print("    2. 在 MSYS2 MinGW64 终端运行:")
            print("         pacman -S mingw-w64-x86_64-gcc")
            print("    3. 重新运行: python build.py")
            print("  或设置环境变量指向 gcc.exe 所在目录:")
            print("    set MINGW_BIN=C:\\path\\to\\mingw64\\bin")
            sys.exit(1)

        print(f"[build] MinGW gcc: {mingw_bin}")
        env = os.environ.copy()
        env["PATH"] = mingw_bin + os.pathsep + env.get("PATH", "")
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace", "--compiler=mingw32"]

    else:  # Linux / macOS
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

    print(f"[build] 运行: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\n[build] 编译失败 (exit {result.returncode})")
        if plat == "Windows":
            print("  提示: 确保 MSYS2 MinGW64 gcc 已正确安装")
            print("        pacman -S mingw-w64-x86_64-gcc")
        elif plat == "Linux":
            print("  提示: 安装依赖")
            print("        sudo apt install build-essential python3-dev")
        elif plat == "Darwin":
            print("  提示: 安装 Xcode Command Line Tools")
            print("        xcode-select --install")
        sys.exit(result.returncode)

    print()
    ok = _verify()
    if ok:
        print(f"\n[build] 成功！_features 模块已就绪，estimate() 将自动使用 C 扩展后端。")
    else:
        print(f"\n[build] 编译完成，但验证失败。请检查上方错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
