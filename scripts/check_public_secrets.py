"""Scan tracked/staged files without printing secret values."""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

RULES = {
    "provider_key": rb"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{25,}|AQ\.[A-Za-z0-9_-]{25,})",
    "github_token": rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "music_cookie": rb"(?:MUSIC_U|MUSIC_R_U|NMTID)=[A-Za-z0-9%_-]{20,}",
    "literal_credential": rb"(?i)(?:password|passwd|api[_-]?key|api[_-]?secret|access[_-]?token|secret)\s*[=:]\s*[\x22\x27][A-Za-z0-9+/]{20,}[\x22\x27]",
    "bearer_token": rb"(?i)Bearer\s+[A-Za-z0-9_.-]{24,}",
}
PRIVATE_DIRS = {"client_data", "users_data", "users_temp", "_internal", "__pycache__", "dist", "build", ".codex_tmp"}
PRIVATE_NAMES = {"credentials.json", "cookie.json", "cookie.txt", "cookies.json", "dialogue_history.txt", "summary_history.txt", "chat_history.txt", "users.json"}

def staged_files(root):
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).split(b"\0")
    paths, ids = [], []
    for entry in filter(None, entries):
        header, path = entry.split(b"\t", 1)
        _, oid, stage = header.split()
        if stage != b"0":
            raise RuntimeError("Resolve Git conflicts before scanning")
        paths.append(path.decode("utf-8"))
        ids.append(oid)
    # Read only Git-index blobs; do not trust a possibly different working copy.
    with tempfile.TemporaryFile() as output:
        subprocess.run(["git", "cat-file", "--batch"], cwd=root,
                       input=b"\n".join(ids)+b"\n", stdout=output, check=True)
        output.seek(0)
        for path in paths:
            _, kind, size = output.readline().split()
            if kind != b"blob":
                raise RuntimeError("Unsupported staged object: " + path)
            content = output.read(int(size))
            if len(content) != int(size) or output.read(1) != b"\n":
                raise RuntimeError("Incomplete Git object: " + path)
            yield path, content

def main():
    asset_dirs = {'model_data', 'voice_refs', 'tts_refs', 'prefab_audio', 'init_musics', 'thumbnails'}
    asset_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.mp3', '.wav', '.ogg', '.flac', '.mp4', '.ttf', '.otf', '.moc3', '.can3', '.asset', '.prefab', '.anim', '.controller', '.meta'}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Read staged blobs, not working-tree files")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    failures = []
    count = 0
    source = staged_files(root) if args.staged else ((p, (root/p).read_bytes()) for p in filter(None, paths))
    for relative, content in source:
        path = Path(relative)
        if asset_dirs.intersection(path.parts) or path.suffix.lower() in asset_extensions:
            failures.append((relative, 'material asset excluded from source-only repository'))
        if any(part in PRIVATE_DIRS or part.startswith((".sync-backup", ".history", ".backup")) for part in path.parts) or path.name in PRIVATE_NAMES or path.suffix in {".log", ".pem", ".key", ".pyc", ".db", ".exe", ".zip"} or ".bak" in path.name or (path.name.startswith(".env") and path.name != ".env.example"):
            failures.append((relative, "private/generated path"))
        for label, pattern in RULES.items():
            if re.search(pattern, content):
                failures.append((relative, label))
        count += 1
    for relative, reason in failures:
        print(f"FAIL {relative}: {reason}")
    print(f"Checked {count} files; findings={len(failures)}")
    raise SystemExit(bool(failures))

if __name__ == "__main__":
    main()

