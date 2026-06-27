# SSH Access to the Windows GPU Server (gtpk@192.168.0.9)

Document position: [Index](./index.md). Practical guide for connecting to and running commands
on the Windows dev/GPU box over SSH — including the exact key-auth setup, cmd.exe quoting
gotchas, long-job pattern, and the errors we actually hit (with fixes). For the full machine
spec + dev-env state, see [Windows Dev Environment](./windows_dev_environment.md).

```text
host : 192.168.0.9   (LAN)
user : gtpk
auth : public-key, passwordless (no password auth from a non-interactive agent shell)
shell: Windows cmd.exe (Windows 11). PowerShell available via `powershell -Command "..."`.
```

---

## 1. Quick connect (after key is installed)

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "hostname & ver"
```

`-o BatchMode=yes` makes ssh fail fast instead of hanging on a password prompt (the agent's
shell is non-interactive). One-shot: every `ssh ... "<cmd>"` is a fresh cmd.exe session — no
state (cwd, env) persists between calls, so `cd` into the repo inside each command.

## 2. One-time key setup (how passwordless was enabled)

The agent (Mac dev box) uses `~/.ssh/id_ed25519`. Its public key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFOPVhUslnm/O8UZAxvxhRaeT0K+VPgRnnPR7r0L0nNE claude-code-bitnet
```

Generate one on a new client if needed: `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519`.

Install the public key **on the Windows server** (PowerShell as gtpk). Pick the right file:

**Normal (non-admin) user** -> per-user `authorized_keys`:

```powershell
mkdir -Force $env:USERPROFILE\.ssh
Add-Content $env:USERPROFILE\.ssh\authorized_keys 'ssh-ed25519 AAAA... claude-code-bitnet'
icacls $env:USERPROFILE\.ssh\authorized_keys /inheritance:r /grant:r "$($env:USERNAME):F"
```

**Administrator user** -> Windows OpenSSH ignores per-user keys for admins and reads ONE shared
file instead:

```powershell
Add-Content C:\ProgramData\ssh\administrators_authorized_keys 'ssh-ed25519 AAAA... claude-code-bitnet'
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant Administrators:F /grant SYSTEM:F
```

Permissions matter: if `authorized_keys` is world-writable, sshd silently refuses it (you'll
keep getting "Permission denied"). The `icacls ... /inheritance:r` lines fix that.

## 3. Enabling the SSH server (if connect times out)

If `ssh` gives `connect to host ... port 22: Operation timed out`, the OpenSSH **server** isn't
running/reachable. On the Windows box (PowerShell as admin):

```powershell
# install (if missing) + start + auto-start
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
# open the firewall for port 22 (some setups need this explicitly)
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH SSH Server' -Enabled True `
  -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

Confirm the box's LAN IP with `ipconfig` (IPv4 of the active adapter); make sure it matches
`192.168.0.9`. Default port is 22.

## 4. Running commands (cmd.exe quoting)

The remote shell is **cmd.exe**, not bash. Inside the double-quoted `ssh "..."` payload:

```text
- separate statements with  &   (sequential)  or  &&  (stop on failure)
- environment variables:    %VAR%   (e.g. %errorlevel%, %USERPROFILE%)
- paths use backslashes:     C:\Users\gtpk\...   (forward slashes also work in most tools)
- list files:   dir /b <path>     find a path:  where <exe>     search text:  findstr /I "needle"
- check exit:   ... & echo EXIT=%errorlevel%
- run PowerShell for richer logic:  powershell -Command "Get-Service sshd"
```

Use the project's conda env python explicitly (no `conda activate` in a one-shot shell):

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 \
  "cd C:\Users\gtpk\BitNet-Transformers & C:\Users\gtpk\anaconda3\envs\bnt\python.exe scripts\<driver>.py <args>"
```

Output buffering note: chaining a remote `python -c "print(...)"` with `& echo ...` sometimes
swallows the python stdout while still returning the right exit code. If you need to SEE output
reliably, write it to a file and `type` it:
`python -c "open('out.txt','w').write(str(x))" & type out.txt`.

## 5. Long-running jobs

Each `ssh "..."` is one-shot and ends when the command returns. Two patterns, by job length.
**Important (tested):** inline detach via cmd `start /b`, PowerShell `Start-Process`, or
`schtasks /tr "python -c ..."` all break on the nested ssh -> cmd -> (powershell) -> python
quoting (you get "path not found" and nothing runs). Don't fight the quoting — use one of these.

### A. Short/medium jobs (probes, inference; minutes) — foreground + redirect, VERIFIED

Run the python in the **foreground** with output redirected to a logfile, and background the
*ssh call itself* on the client side (the agent's Bash `run_in_background: true`). The SSH
session holds the remote process for its duration; on a LAN this is reliable for ~tens of
minutes. Poll the log on separate ssh calls.

```bash
# launch (client-side background; the remote python runs in the foreground of this ssh session)
ssh -o BatchMode=yes gtpk@192.168.0.9 \
  "cd C:\Users\gtpk\BitNet-Transformers & C:\Users\gtpk\anaconda3\envs\bnt\python.exe scripts\<driver>.py <args> > C:\Users\gtpk\run.log 2>&1"
# poll on another call
ssh -o BatchMode=yes gtpk@192.168.0.9 "type C:\Users\gtpk\run.log" 2>&1 | tail
ssh -o BatchMode=yes gtpk@192.168.0.9 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

### B. Truly long / must-survive-disconnect — a committed .bat launcher

For multi-hour jobs that must outlive the ssh connection, put the command in a **real .bat
file** (no quoting hell, since it's a file on disk), then launch it detached. The .bat is the
only reliable way to get nested python args past cmd. Pattern:

```bat
:: C:\Users\gtpk\BitNet-Transformers\run_job.bat   (commit a template or write via PowerShell Set-Content)
cd /d C:\Users\gtpk\BitNet-Transformers
C:\Users\gtpk\anaconda3\envs\bnt\python.exe scripts\<driver>.py <args> > C:\Users\gtpk\run.log 2>&1
```

```bash
# launch the .bat detached and persistent (window hidden, survives logout):
ssh -o BatchMode=yes gtpk@192.168.0.9 \
  "schtasks /create /tn bntjob /tr C:\Users\gtpk\BitNet-Transformers\run_job.bat /sc once /st 00:00 /f & schtasks /run /tn bntjob"
# status / cleanup
ssh -o BatchMode=yes gtpk@192.168.0.9 "tasklist /fi \"imagename eq python.exe\" /fo csv | findstr python"
ssh -o BatchMode=yes gtpk@192.168.0.9 "schtasks /delete /tn bntjob /f"
```

(schtasks pointing at a **.bat file path** works — only inline `python -c` inside `/tr` fails.)

This box is **persistent** (unlike Colab's recycling VMs); just `git push` results to origin
from the box when done (git installed, repo is a clone of gtpk's GitHub) — no Drive/base64 relay.
Note: the heavy FACT *training* (~17 GB) does not fit the 3080's 10 GB; those run on Colab L4.
The 3080 is for probes/inference/dev, where pattern A is enough.

## 6. Troubleshooting (errors we hit, with fixes)

| symptom | cause | fix |
| --- | --- | --- |
| `port 22: Operation timed out` | sshd not running / firewall / wrong IP | start sshd + firewall rule (§3); verify `ipconfig` |
| `Permission denied (publickey,password,...)` | key not installed, or wrong file (admin vs user), or bad perms | install pubkey in the correct authorized_keys (§2); fix `icacls` |
| ssh hangs | password prompt with no TTY | add `-o BatchMode=yes`; finish key-auth setup |
| remote stdout missing but EXIT=0 | cmd `&`-chaining buffering | write to a file and `type` it (§4) |
| `ModuleNotFoundError` | used base/system python, not the env | call `...\envs\bnt\python.exe` explicitly (§4) |
| host key changed warning | reinstall / new host | `ssh-keygen -R 192.168.0.9` then reconnect |
