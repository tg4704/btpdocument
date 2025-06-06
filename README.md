The power grid is changing fast with more renewable energy sources and smart devices being added to distribution systems. We're seeing solar panels, electric car chargers, and smart appliances everywhere, making the distribution side much more active than it used to be.

In the past, we could model distribution systems as simple "lump loads" - basically treating them like one big passive consumer. This worked fine when most loads just consumed power in predictable ways. But now with Distributed Energy Resources (DERs) like solar panels and battery storage, power can flow both ways and loads behave much more dynamically.

So the big question is: can we still use the old lumped load models, or do we need something more sophisticated? This study looks at whether our traditional modeling approach is still good enough or if we're missing important details by oversimplifying things.
This matters because accurate models are crucial for planning and operating the grid effectively. Getting this right will help us better integrate all the new renewable technologies coming online.


# Python Environment Setup Guide for Jupyter in VS Code

## Step 1: Install Python
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Verify installation by opening Command Prompt/Terminal and typing:
   ```
   python --version
   ```

## Step 2: Install VS Code
1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install it with default settings

## Step 3: Install VS Code Extensions
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Install these extensions:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)
   - **Jupyter Keymap** (optional, for familiar shortcuts)

## Step 4: Create a Virtual Environment (Recommended)
1. Open VS Code
2. Open Terminal in VS Code (Ctrl+`)
3. Navigate to your project folder:
   ```
   cd path/to/your/project
   ```
4. Create virtual environment:
   ```
   python -m venv myenv
   ```
5. Activate it:
   - **Windows:** `myenv\Scripts\activate`
   - **Mac/Linux:** `source myenv/bin/activate`

## Step 5: Install Required Libraries
Copy and paste this command in your terminal (make sure virtual environment is activated):

```bash
pip install pypower matplotlib numpy scipy jupyter ipykernel
```

If you get errors with pypower, try:
```bash
pip install PYPOWER
```

## Step 6: Set Up Jupyter in VS Code
1. Create a new file with `.ipynb` extension (e.g., `power_analysis.ipynb`)
2. VS Code will automatically detect it as a Jupyter notebook
3. Select your Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose the one from your virtual environment (should show the path to myenv)

## Step 7: Test Your Setup
Create a new cell in your notebook and run:

```python
# Test imports
from pypower.api import case9, runpf, printpf, case14
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import copy, csv
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)

print("All libraries imported successfully!")
print("NumPy version:", np.__version__)
```

## Step 8: Running Your Code
- Click on a cell and press `Shift+Enter` to run it
- Use `Ctrl+Enter` to run without moving to next cell
- Add new cells with the `+` button

