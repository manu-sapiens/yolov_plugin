if exist ".\venv" (
	call .\venv\Scripts\activate
	python -m ensurepip --upgrade
	python -m pip install --upgrade "pip>=22.3.1,<23.1.*"
	python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
	python -m pip install -r requirements.txt

	call download_file.bat "https://github.com/ultralytics/assets/releases/download/v0.0.0" yolov8n-seg.pt checkpoints
	call download_file.bat "https://github.com/ultralytics/assets/releases/download/v0.0.0" yolov8s-seg.pt checkpoints
	call download_file.bat "https://github.com/ultralytics/assets/releases/download/v0.0.0" yolov8m-seg.pt checkpoints
	call download_file.bat "https://github.com/ultralytics/assets/releases/download/v0.0.0" yolov8l-seg.pt checkpoints
	call download_file.bat "https://github.com/ultralytics/assets/releases/download/v0.0.0" yolov8x-seg.pt checkpoints

)
