import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "plugins" / "otzaria-ai"


def copy_plugin_tree(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        PLUGIN_DIR,
        target,
        ignore=shutil.ignore_patterns(
            ".DS_Store",
            "__MACOSX",
            "*.map",
            "node_modules",
            ".git",
        ),
    )


def update_manifest_version(staging_dir: Path, version: str) -> None:
    manifest_path = staging_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = version
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_plugin(staging_dir: Path) -> None:
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json is missing from plugin root")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entrypoint = manifest.get("entrypoint")
    if not entrypoint or not (staging_dir / entrypoint).exists():
        raise FileNotFoundError(f"Plugin entrypoint is missing: {entrypoint}")


def write_archive(staging_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_dir).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Otzaria AI .otzplugin file.")
    parser.add_argument("--version", default=None, help="Override manifest version.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "dist" / "OtzariaAI.otzplugin"),
        help="Output .otzplugin path.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(tmp) / "plugin"
        copy_plugin_tree(staging_dir)
        if args.version:
            update_manifest_version(staging_dir, args.version)
        validate_plugin(staging_dir)
        write_archive(staging_dir, Path(args.output).resolve())

    print(f"Created {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
