# Pair Analysis — Setup Guide

A web app for comparing two stocks statistically (correlation, R², beta, and more).

---

## Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python"** button
3. Run the installer
   - ✅ **Important:** On the first screen, check the box that says **"Add Python to PATH"**
4. Click **Install Now** and follow the prompts

To verify it worked, open **Terminal** (Mac) or **Command Prompt** (Windows) and type:
```
python --version
```
You should see something like `Python 3.12.x`

---

## Step 2 — Download the project

Save the `pair-analysis` folder somewhere easy to find, like your Desktop or Documents.

---

## Step 3 — Open a terminal in the project folder

**On Mac:**
1. Open the **Terminal** app (search for it in Spotlight with `Cmd + Space`)
2. Type `cd ` (with a space after), then drag the `pair-analysis` folder into the terminal window and press Enter

**On Windows:**
1. Open the `pair-analysis` folder in File Explorer
2. Click the address bar at the top, type `cmd`, and press Enter

---

## Step 4 — Install the required packages

Copy and paste this command into your terminal and press Enter:

```
pip install -r requirements.txt
```

This installs Flask (the web server), yfinance (Yahoo Finance data), and numpy (statistics). It only needs to be done once.

---

## Step 5 — Run the app

```
python app.py
```

You should see:
```
✓ Pair Analysis server running at http://localhost:5000
```

---

## Step 6 — Open the app

Open your browser and go to:

**http://localhost:5000**

Enter two stock tickers (e.g. `AAPL` and `MSFT`), pick a time period, and click **Analyze**.

---

## Stopping the server

When you're done, go back to the terminal and press **Ctrl + C** to stop the server.

## Starting it again next time

Just repeat Step 5 — you won't need to reinstall packages again.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Try `python3` instead of `python` |
| `pip: command not found` | Try `pip3` instead of `pip` |
| Page shows "Is the server running?" | Make sure you ran `python app.py` and see the startup message |
| Ticker not found | Double-check the ticker on Yahoo Finance (https://finance.yahoo.com) |

---

*Data sourced via Yahoo Finance. For informational purposes only — not investment advice.*
