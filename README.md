# ACS Theatre Intelligence Explorer
## Adult Cardiac Surgery — Theatre Operations Analysis & ML Prediction

An interactive Gradio app that turns raw ACS theatre data into plain English insights,
built for NHS data scientist analysts.

---

## Quick Start (VS Code)

### 1. Create a project folder

Create a new folder anywhere on your machine, e.g.:
```
C:\Users\David\Projects\acs-theatre-explorer\
```

### 2. Put both files in the folder

Copy these two files into that folder:
- `acs_theatre_explorer.py` (the app)
- `ACSData_Data_.csv` (your data)

### 3. Open in VS Code

- Open VS Code
- File → Open Folder → select your project folder
- Open a terminal: Terminal → New Terminal (or Ctrl+`)

### 4. Install dependencies

In the VS Code terminal, run:
```bash
pip install gradio pandas scikit-learn plotly
```

If you use a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install gradio pandas scikit-learn plotly
```

### 5. Run the app

```bash
python acs_theatre_explorer.py
```

### 6. Open your browser

The terminal will show:
```
Running on local URL: http://127.0.0.1:7860
```

Click that link or paste it into your browser. That's it — you're in.

---

## What's Inside

| Tab | What It Does |
|-----|-------------|
| 📊 Overview | Plain English summary of the entire dataset |
| 🏥 Theatre Efficiency | Scheduled vs actual durations, overrun analysis |
| 👨‍⚕️ Surgeon Profiles | Workload, case mix, and timing by surgeon |
| 📅 Scheduling | Day-of-week and monthly patterns |
| ❌ Cancellations | What's getting cancelled and why |
| 🤖 ML Prediction | Machine learning model + predict new cases |
| 🔍 Data Quality | Missing data assessment and recommendations |

---

## Requirements

- Python 3.9+
- gradio
- pandas
- scikit-learn
- plotly

---

*Built by David — AI Engineer, NHS 365 AI Technologies*
