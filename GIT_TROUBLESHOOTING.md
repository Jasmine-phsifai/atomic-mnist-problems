# Git Troubleshooting

Local-environment notes for this repo. Read this first when `git pull` / `git push` /
`git ls-remote` fails.

## Symptom: `fatal: unable to access ... Connection was reset`

Also appears as `Failed to connect to github.com port 443`, `Recv failure`,
`OpenSSL SSL_read: Connection was reset`, or a `git push` that hangs then dies.

### Root cause

GitHub is reached through a local proxy client (Clash). Windows stores that proxy in
the WinINET system settings, but **Git does not read WinINET settings**. So the browser
works while Git talks to GitHub directly and gets reset.

### Diagnosis, in order

```powershell
# 1. Is it actually the network, or a Git/auth error?
git ls-remote origin

# 2. What does Windows think the proxy is?
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' |
    Select-Object ProxyEnable, ProxyServer, AutoConfigURL

# 3. Is the proxy client actually running?
Get-Process | Where-Object { $_.ProcessName -match 'clash|mihomo|v2ray|xray|sing-box|verge' }

# 4. Is the port actually listening?
Test-NetConnection -ComputerName 127.0.0.1 -Port 10080 -InformationLevel Quiet

# 5. Does Git work when told about the proxy?
git -c http.proxy=http://127.0.0.1:10080 ls-remote origin
```

If step 5 succeeds and step 1 failed, the proxy config is the whole problem.

### Fix

Scope the proxy to GitHub only, so other remotes (internal mirrors, local paths) are
untouched:

```powershell
git config --global http.https://github.com.proxy http://127.0.0.1:10080
git config --global https.https://github.com.proxy http://127.0.0.1:10080
```

Verify:

```powershell
git ls-remote origin
```

### Current machine values

- Proxy client: Clash
- Mixed HTTP/SOCKS port: `127.0.0.1:10080` (matches the Windows system proxy setting)
- If Clash's port changes, update both `git config` keys above to the new port.

### If the port changed

```powershell
# find what clash is listening on
Get-NetTCPConnection -State Listen -OwningProcess (Get-Process clash).Id |
    Select-Object LocalAddress, LocalPort
```

Then re-run the two `git config` commands with the new port.

### Removing the proxy config

```powershell
git config --global --unset http.https://github.com.proxy
git config --global --unset https.https://github.com.proxy
```

## Symptom: DNS lookup for github.com returns nothing

`Resolve-DnsName github.com` failing is expected when the proxy client runs in TUN /
fake-IP mode or hijacks DNS. It is **not** a reliable health check — use
`git ls-remote origin` instead.

## Symptom: push rejected, `non-fast-forward`

Someone (you, in another session) pushed first.

```powershell
git fetch origin
git rebase origin/main     # or: git pull --rebase
```

Never `git pull` with a merge across a rewritten history — a merge resurrects deleted
content. After any force-push or history rewrite upstream:

```powershell
git fetch origin
git reset --hard origin/main
```

## Symptom: authentication prompt loops / `could not read Username`

Credentials live in Windows Credential Manager via Git Credential Manager.

```powershell
git config --global credential.helper        # should print: manager
cmdkey /list:git:https://github.com          # inspect stored credential
```

To reset a stale token, remove the `git:https://github.com` entry from
*Windows Credential Manager → Windows Credentials*, then run any remote command and
re-authenticate.

## Symptom: `RPC failed; curl 92 HTTP/2 stream ... was not closed cleanly`

Large push over a flaky tunnel. Force HTTP/1.1 and raise the buffer:

```powershell
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```

## Symptom: whole files show as modified with no visible change

Line-ending normalization. This repo's `.gitattributes` sets `* text=auto eol=lf`.

```powershell
git add --renormalize .
```

## Quick health check

```powershell
git remote -v
git ls-remote origin
git status -sb
```

If all three are clean, the network side is fine and the problem is elsewhere.
