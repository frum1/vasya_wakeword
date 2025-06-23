import os
import sys
import hashlib
import argparse

def md5_hash(text: str) -> str:
    """Возвращает hex-строку MD5-хэша от text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def rename_files(directory: str, recursive: bool = False):
    """
    Переименовывает все файлы в directory.
    Если recursive=True, то заходит в поддиректории.
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            name, ext = os.path.splitext(filename)
            new_name = md5_hash(name) + ext
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_name)
            # Проверяем, не совпадает ли уже
            if old_path != new_path:
                if os.path.exists(new_path):
                    print(f"WARNING: {new_path} уже существует, пропускаем {old_path}")
                else:
                    os.rename(old_path, new_path)
                    print(f"{old_path} -> {new_path}")
        if not recursive:
            # если не рекурсивно, выходим сразу после первого каталога
            break

def main():
    parser = argparse.ArgumentParser(
        description="Переименовывает файлы, заменяя имена на MD5-хэши от исходного имени (расширения не трогаются)."
    )
    parser.add_argument(
        "directory",
        help="Папка, в которой нужно переименовать файлы"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Заходить в поддиректории (рекурсивно). По умолчанию: только текущая папка."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"ERROR: {args.directory} не является директорией.")
        sys.exit(1)

    rename_files(args.directory, args.recursive)

if __name__ == "__main__":
    main()
