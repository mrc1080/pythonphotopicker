cd $PSScriptRoot

if (!(Test-Path ".\.venv")) {
    py -3.12 -m venv .venv
}

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
if (Test-Path ".\requirements.txt") {
    python -m pip install -r requirements.txt
} else {
    python -m pip install opencv-python pillow imagehash numpy tqdm
}

python -m pip install pyinstaller
python -m PyInstaller PhotoPicker.spec
