# Windows Server Deployment

This file is for deploying the backend in `server/` on a Windows machine.
It intentionally excludes desktop client packages such as `PySide6`, `pygame`, and `live2d`.

## 1. Recommended environment

- Windows Server or Windows 10/11
- Python 3.10 or 3.11, 64-bit
- PowerShell

## 2. Create a virtual environment

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Install server dependencies

```powershell
pip install -r requirements-server.txt
```

## 4. Start the server

```powershell
python server\run.py
```

If startup succeeds, the server listens on:

```text
http://0.0.0.0:5000
```

Clients on other machines should connect to:

```text
http://<your-server-ip>:5000
```

## 5. Open the firewall port

If other devices need to access this service, allow inbound TCP `5000` in Windows Firewall.

## 6. Notes for this project

- `requirements-server.txt` includes only packages imported by the backend.
- The NetEase module depends on `pymusiclibrary`.
- The speech translation module depends on `websocket-client`.
- Your current `server/chat_ai.py` and `server/translator.py` contain hard-coded keys. Move them to environment variables before putting this on a real server.
- Your current `server/run.py` runs with `debug=True`. For a real server, change that to `debug=False`.

## 7. Minimal production tweak

In `server/run.py`, change:

```python
socketio.run(app, host="0.0.0.0", port=5000, debug=True)
```

to:

```python
socketio.run(app, host="0.0.0.0", port=5000, debug=False)
```

## 8. Optional smoke test

On the server itself:

```powershell
Invoke-WebRequest http://127.0.0.1:5000
```

If your routes are up, you should get an HTTP response instead of a connection error.
