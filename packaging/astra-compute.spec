from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

root = Path(SPECPATH).parent
analysis = Analysis(
    [str(root / "python/astra/worker.py")],
    pathex=[str(root / "python")],
    binaries=collect_dynamic_libs("mlx"),
    datas=collect_data_files("mlx"),
    # Native MLX initialization imports Python helpers (including
    # mlx._reprlib_fix) that static Python import analysis cannot discover.
    hiddenimports=collect_submodules("mlx", filter=lambda name: name not in {"mlx.extension", "mlx.__main__"}) + ["numpy"],
    excludes=["torch", "torchvision", "tensorflow", "jax", "pytest"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive, analysis.scripts, [], exclude_binaries=True, name="AstraCompute",
    debug=False, strip=False, upx=False, console=True,
    # None selects PyInstaller's ordinary ad-hoc signing. Passing the literal
    # '-' is treated as a real identity and enables hardened runtime, whose
    # team-ID library validation cannot load an ad-hoc Python distribution.
    target_arch="arm64", codesign_identity=None,
)
collection = COLLECT(executable, analysis.binaries, analysis.datas, strip=False, upx=False, name="AstraCompute")
app = BUNDLE(
    collection, name="AstraCompute.app", bundle_identifier="com.unendless.agenttrainer.astra.compute",
    info_plist={"CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "1", "LSMinimumSystemVersion": "15.0", "LSBackgroundOnly": True},
)
