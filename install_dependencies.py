import subprocess
import sys

def install_requirements():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

if __name__ == "__main__":
    install_requirements()