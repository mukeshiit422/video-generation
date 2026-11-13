"""
Patches numpy 2.x incompatibilities in third-party packages.
Run once after install: python patch_dependencies.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent

REPLACEMENTS = [
    # numpy 2.x removed these type aliases
    (r'\bnp\.bool8\b',                  'np.bool_'),
    (r'\bnp\.int0\b',                   'np.intp'),
    (r'\bnp\.uint0\b',                  'np.uintp'),
    (r'\bnp\.bool\b',                   'bool'),
    (r'\bnp\.int\b(?!\w)',              'int'),
    (r'\bnp\.float\b(?!\w)',            'float'),
    (r'\bnp\.complex\b(?!\w)',          'complex'),
    (r'\bnp\.object\b(?!\w)',           'object'),
    (r'\bnp\.str\b(?!\w)',              'str'),
    # VisibleDeprecationWarning moved to numpy.exceptions in 2.x
    (r'\bnp\.VisibleDeprecationWarning\b', 'DeprecationWarning'),
    # functional_tensor removed in torchvision 0.16+
    (
        r'from torchvision\.transforms\.functional_tensor import rgb_to_grayscale',
        'from torchvision.transforms.functional import rgb_to_grayscale',
    ),
    # SadTalker uses Python 2 raise syntax
    (
        r"raise 'can not detect the landmark from source image'",
        "raise RuntimeError('can not detect the landmark from source image')",
    ),
    # SadTalker POS(): b is shape (2n,1) so lstsq returns k shape (8,1).
    # k[3] is shape (1,) and float(shape-(1,)-array) is a hard error in numpy 2.x.
    # Flatten k to 1D immediately so all indexing yields proper scalars.
    (
        r'k, _, _, _ = np\.linalg\.lstsq\(A, b\)\n(?!\s*k = k\.ravel\(\))',
        'k, _, _, _ = np.linalg.lstsq(A, b)\n    k = k.ravel()  # (8,1) → (8,) so k[n] is scalar\n',
    ),
    # GFPGAN downloads facexlib weights to a cwd-relative 'gfpgan/weights'. Run from
    # the project root, that creates ./gfpgan/ which shadows the gfpgan package
    # ("cannot import name 'GFPGANer' from 'gfpgan' (unknown location)").
    # Anchor it to the GFPGAN repo dir instead.
    (
        r"model_rootpath='gfpgan/weights'\)",
        "model_rootpath=os.path.join(ROOT_DIR, 'gfpgan/weights'))",
    ),
    # SadTalker loads facexlib weights from a cwd-relative 'gfpgan/weights'. Any gfpgan/
    # dir in its cwd (third_party/SadTalker) shadows the gfpgan package and SadTalker dies
    # on import. Read the location from an env var instead (face_animator sets it to
    # models/gfpgan/), so no gfpgan/ directory is ever needed.
    (
        r"root_path = 'gfpgan/weights'",
        "root_path = os.environ.get('SADTALKER_FACEXLIB_DIR', 'gfpgan/weights')",
    ),
    # torchvision 0.13+ deprecated pretrained=False (facexlib retinaface); weights=None is identical
    (r'models\.resnet50\(pretrained=False\)', 'models.resnet50(weights=None)'),
    # Also replace the old np.stack form if it survived a previous patch run
    (
        r't = np\.stack\(\[sTx, sTy\], axis=0\)',
        't = np.array([float(sTx), float(sTy)])',
    ),
]

PATCH_DIRS = [
    ROOT / 'third_party' / 'SadTalker',
    ROOT / 'third_party' / 'GFPGAN',
    ROOT / 'third_party' / 'Real-ESRGAN',
]

VENV_PATCH_FILES = [
    ROOT / 'venv/lib/python3.11/site-packages/basicsr/data/degradations.py',
    ROOT / 'venv/lib/python3.11/site-packages/skimage/util/dtype.py',
    ROOT / 'venv/lib/python3.11/site-packages/facexlib/detection/retinaface.py',
]


def patch_file(path: Path) -> int:
    try:
        original = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return 0

    patched = original
    for pattern, replacement in REPLACEMENTS:
        patched = re.sub(pattern, replacement, patched)

    if patched != original:
        path.write_text(patched, encoding='utf-8')
        changes = sum(1 for p, r in REPLACEMENTS if re.search(p, original))
        print(f'  PATCHED  {path.relative_to(ROOT)}  ({changes} fix(es))')
        return 1
    return 0


def main():
    print('Patching numpy 2.x / torchvision incompatibilities...\n')
    total = 0

    # Patch third-party repos
    for d in PATCH_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*.py'):
            total += patch_file(f)

    # Patch specific venv files
    for f in VENV_PATCH_FILES:
        if f.exists():
            total += patch_file(f)

    # skimage bool8 is in a .py so patch it
    skimage_dtype = ROOT / 'venv/lib/python3.11/site-packages/skimage/util/dtype.py'
    if skimage_dtype.exists():
        total += patch_file(skimage_dtype)

    print(f'\nDone — {total} file(s) patched.')


if __name__ == '__main__':
    main()
