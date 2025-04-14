#!/bin/bash
echo "Installing Multi-scale Adaptive Network (MAN) Epidemic Model..."
echo

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv man_epidemic_env
source man_epidemic_env/bin/activate

# Install required packages
echo "Installing required packages (this may take few minutes)..."
pip install numpy pandas networkx matplotlib scipy seaborn tqdm

# Copy application files
echo "Copying application files..."
mkdir -p man_epidemic_app
cp man_epidemic_model.py man_epidemic_app/
cp man_epidemic_gui.py man_epidemic_app/

# Create launcher
echo "Creating launcher..."
cat > man_epidemic.command << EOL
#!/bin/bash
cd "\$(dirname "\$0")"
source man_epidemic_env/bin/activate
cd man_epidemic_app
python3 man_epidemic_gui.py
EOL

chmod +x man_epidemic.command

echo
echo "Installation complete!"
echo "Run man_epidemic.command to start the application"
