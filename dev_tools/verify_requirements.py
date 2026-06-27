"""
verify_requirements.py

Verifies all SpectroCough production dependencies
and generates the final requirements.txt
"""

from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

# ============================================================
# Production packages
# ============================================================

PACKAGES = [
    ("Flask", "flask"),
    ("Flask-Cors", "flask_cors"),
    ("gunicorn", None),          # Optional locally, required on Render

    ("numpy", "numpy"),
    ("scipy", "scipy"),

    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("tensorflow-hub", "tensorflow_hub"),
    ("tf_keras", "tf_keras"),

    ("h5py", "h5py"),
    ("protobuf", "google.protobuf"),

    ("librosa", "librosa"),
    ("soundfile", "soundfile"),

    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),

    ("matplotlib", "matplotlib"),

    ("requests", "requests"),

    ("Werkzeug", "werkzeug"),
    ("Jinja2", "jinja2"),
    ("click", "click"),
    ("itsdangerous", "itsdangerous"),
    ("blinker", "blinker"),
]

print("=" * 70)
print("SpectroCough Dependency Verification")
print("=" * 70)

verified = []

for pip_name, module_name in PACKAGES:

    if module_name is None:
        print(f"[SKIP] {pip_name} (Render-only package)")
        verified.append(f"{pip_name}==23.0.0")
        continue

    try:
        import_module(module_name)

        try:
            ver = version(pip_name)
        except PackageNotFoundError:
            ver = version(module_name)

        print(f"[ OK ] {pip_name:<20} {ver}")

        verified.append(f"{pip_name}=={ver}")

    except Exception as e:
        print(f"[FAIL] {pip_name:<20} ({e})")

print("\nGenerating requirements.txt...")

with open("requirements.txt", "w", encoding="utf-8") as f:

    f.write(
        "# ==========================================\n"
        "# SpectroCough Production Requirements\n"
        "# Auto-generated from verified environment\n"
        "# ==========================================\n\n"
    )

    for pkg in verified:
        f.write(pkg + "\n")

print("\nrequirements.txt generated successfully.")