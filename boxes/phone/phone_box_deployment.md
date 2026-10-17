# Running the phone box: step by step

The phone box lets a smartphone act as a camera and annotation terminal for your Python program. The person holding the phone opens a web page, takes a photo, taps some labels and sends. The photos wait inside the box. Your program asks the box for them with ordinary `run()` calls, and sends results back the same way.

Two things talk to the box, on two different ports:

| What | Port in the container | Who uses it |
|---|---|---|
| gRPC | 8061 | Your Python program (no password, keep it private) |
| Web page | 8062 | The phone's browser |

Both guides below publish them on the host as **9079** (gRPC) and **9080** (web). Section 1 does it on your own computer, with the phone on the same Wi-Fi or LAN. Section 2 does it on a server on the internet.

One rule explains most of the setup: **the phone's camera only works on an `https://` page.** The box can serve HTTPS itself, which is why both sections deal with a certificate.

---

## 1. On your own computer, phone on the same LAN

You need Docker on your computer, a phone on the same network (same Wi-Fi), and Python with `visionist_client` installed.

### 1.1 Find your computer's address on the LAN

The phone needs your computer's LAN address, something like `192.168.1.20`. Do not use `localhost`, because on the phone that means the phone itself.

- **Linux:** `hostname -I` (take the first address that looks like `192.168.x.x` or `10.x.x.x`)
- **macOS:** `ipconfig getifaddr en0` (try `en1` if it prints nothing)
- **Windows:** run `ipconfig` and read "IPv4 Address" under your Wi-Fi or Ethernet adapter

Write it down. In this guide it is `192.168.1.20`. Replace it with yours everywhere.

### 1.2 Start the container

```bash
docker run -d --name phone --restart unless-stopped \
  -p 9079:8061 \
  -p 9080:8062 \
  -e TLS_SELF_SIGNED=1 \
  -e TLS_SAN=192.168.1.20 \
  -e PUBLIC_URL=https://192.168.1.20:9080 \
  docker.io/sipgisr/visionist-phone:latest
```

What this does:

- `-p 9079:8061` makes the gRPC port reachable at `localhost:9079` on your computer.
- `-p 9080:8062` makes the web page reachable at port 9080.
- `TLS_SELF_SIGNED=1` makes the box create its own throw-away certificate, so the page is served over `https://` without you getting one. `TLS_SAN` puts your LAN address into that certificate.
- `PUBLIC_URL` is the address the box will put in the pairing link and QR code. It must be an address the phone can reach.

Check that it started:

```bash
docker logs phone
```

You should see a line like `web server on 0.0.0.0:8062 (https)` and `Server started at [::]:8061`. If your computer has a firewall, allow incoming connections on port 9080. If your LAN address changes later (for example after a router restart), stop the container with `docker rm -f phone` and start it again with the new address.

### 1.3 Create the pairing and get the token

Nothing can be paired until your program asks the box to set up a session. In Python:

```python
from visionist_client import Visionist

phone = Visionist("localhost:9079")
r = phone.run(config={"phone": {
    "command": "setup",
    "title": "Annotate",
    "labels": ["handle", "chip", "ok"],     # the buttons shown on the phone
}})

info = r.config["phone"]
print(info["url"])                           # the pairing link, with the token in it
print(info.get("warnings"))                  # anything the box wants you to know

with open("pair.png", "wb") as f:            # the same link as a QR code
    f.write(r.fields["qr"])
```

The **token** is the long random text after `#t=` in the link. The link looks like this:

```
https://192.168.1.20:9080/#t=Qm9u...long-random-text...
```

The link is the only thing the phone needs. Treat it like a password: anyone who has it can send photos to your session and see your results. To invalidate an old link and get a fresh one, call `setup` again with `"rotate": True`. Phones paired with the old link are then refused.

If `info["url"]` is empty or has no address, the box doesn't know its public address. Add `"public_url": "https://192.168.1.20:9080"` to the `setup` parameters.

### 1.4 Open it on the phone

Either open `pair.png` and scan it with the phone's camera app, or type the link into the phone's browser exactly as printed, including the `#t=...` part.

The first time, the browser warns that the connection is not private, because the certificate is self-signed. This is expected on your own network. Accept it once:

- **Chrome on Android:** tap **Advanced**, then **Proceed to 192.168.1.20 (unsafe)**
- **Safari on iPhone:** tap **Show Details**, then **visit this website**

The page then shows your title and label buttons, and asks for permission to use the camera. Allow it. If the browser refuses the camera on a self-signed page, the page falls back to the phone's own camera app, which still works but has no live preview.

### 1.5 Take a photo and send it

On the phone: point the camera, tap the labels that apply (several can be on at once), optionally type a note, then tap capture and send. The page confirms when the photo has been stored in the box.

### 1.6 Get the photos in your program

Your program keeps asking the box for anything new. `after` is a bookmark so you don't get the same photo twice, and `wait_s` makes the box hold the request for up to that many seconds until something arrives:

```python
import json

after = 0
while True:
    r = phone.run(config={"phone": {"command": "get",
                                    "after": after,
                                    "wait_s": 15}})
    info = r.config["phone"]
    after = info["next_after"]                       # move the bookmark forward

    meta = r.fields.get("meta") or []
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)
    images = r.fields.get("images") or []

    for item in meta:
        # item has: seq, labels, note, n_images, first_image, device, sensors, ...
        for k in range(item["n_images"]):
            data = images[item["first_image"] + k]
            with open(f"photo_{item['seq']}_{k}.jpg", "wb") as f:
                f.write(data)
        print(item["seq"], item["labels"], item["note"])
```

Each loop either returns new submissions or comes back empty after `wait_s` seconds, so it does not spin. `info["dropped"]` tells you if photos were discarded because you were too slow.

### 1.7 Send a reply to the phone

```python
phone.run(config={"phone": {"command": "show",
                            "title": "Result",
                            "text": "Chip on the rim.",
                            "ref_seq": item["seq"]}})          # shows "Re #<seq>"
```

To send an image or a 3D model as well, pass it as data, for example `data={"image": pathlib.Path("annotated.jpg")}` or `data={"glb": pathlib.Path("map.glb")}`. The phone shows the new result card within a few seconds.

### 1.8 Stop it

```bash
docker rm -f phone
```

Everything lives in memory, so stopping the container discards the session and all unread photos. The phone has to be paired again afterwards.

---

## 2. On a server on the internet

This is the setup from scratch for a server such as `ifetch.isr.tecnico.ulisboa.pt`. The phone box runs on the server. Your Python program runs on a different computer.

You need:

- A server with Docker and `sudo` rights.
- A **domain name** that points to the server, for example `ifetch.isr.tecnico.ulisboa.pt`. A browser will not accept a normal certificate for a bare IP address.
- Ports **80** and **443** reachable from the internet. Ask your network administrator to open them if they are blocked. Port 80 is needed only while getting and renewing the certificate.

### 2.1 Get a certificate

Ask your IT department whether they can issue a certificate for the hostname. If not, use the free Let's Encrypt service:

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d ifetch.isr.tecnico.ulisboa.pt
```

Answer the questions it asks. When it succeeds, the certificate files are in `/etc/letsencrypt/live/ifetch.isr.tecnico.ulisboa.pt/`. If port 80 is already used by a web server on the machine, stop that server for the few seconds certbot needs.

### 2.2 Put the certificate where the container can read it

The container does not run as root, and Let's Encrypt keeps the private key readable by root only. So do not mount `/etc/letsencrypt` directly. Instead, copy the files into a folder of their own and hand the key to the container's user.

First find out which user the container uses:

```bash
docker run --rm --entrypoint id docker.io/sipgisr/visionist-phone:latest
```

It prints something like `uid=100(runner) gid=101(runner-group)`. Note the two numbers (here `100` and `101`).

Now create the folder, fill it, and set the permissions (replace `100:101` with your numbers):

```bash
sudo mkdir -p /opt/phone-certs
D=/etc/letsencrypt/live/ifetch.isr.tecnico.ulisboa.pt
sudo cp -L $D/fullchain.pem $D/privkey.pem /opt/phone-certs/
sudo chmod 755 /opt/phone-certs
sudo chmod 644 /opt/phone-certs/fullchain.pem
sudo chown 100:101 /opt/phone-certs/privkey.pem
sudo chmod 400 /opt/phone-certs/privkey.pem
ls -l /opt/phone-certs
```

The listing should show `fullchain.pem` as `-rw-r--r--` and `privkey.pem` as `-r--------` owned by `100 101`.

The certificate expires after about 90 days, and certbot renews it automatically. This script copies the renewed files to the same place with the same permissions every time:

```bash
sudo tee /etc/letsencrypt/renewal-hooks/deploy/phone.sh >/dev/null <<'EOF'
#!/bin/sh
D=/etc/letsencrypt/live/ifetch.isr.tecnico.ulisboa.pt
UID_GID=100:101
cp -L $D/fullchain.pem $D/privkey.pem /opt/phone-certs/
chmod 644 /opt/phone-certs/fullchain.pem
chown $UID_GID /opt/phone-certs/privkey.pem
chmod 400 /opt/phone-certs/privkey.pem
EOF
sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/phone.sh
```

Use your own numbers for `UID_GID`. The box re-reads the certificate files every hour, so a renewal needs no restart.

### 2.3 Start the container

```bash
docker run -d --name phone --restart unless-stopped \
  -p 443:8062 \
  -p 9079:8061 \
  -v /opt/phone-certs:/certs:ro \
  -e TLS_CERT=/certs/fullchain.pem \
  -e TLS_KEY=/certs/privkey.pem \
  -e PUBLIC_URL=https://ifetch.isr.tecnico.ulisboa.pt \
  -e ALLOWED_HOSTS=ifetch.isr.tecnico.ulisboa.pt \
  docker.io/sipgisr/visionist-phone:latest
```

What this does:

- `-p 443:8062` publishes the phone's web page on the standard HTTPS port, so the link needs no port number.
- `-p 9079:8061` publishes the gRPC port. It has **no password**, so do not leave it open to the whole internet (see 2.5).
- `-v /opt/phone-certs:/certs:ro` makes the certificate folder visible inside the container, read-only.
- `TLS_CERT` and `TLS_KEY` tell the box to serve HTTPS with those files.
- `PUBLIC_URL` is the address the box puts in the pairing link and QR code.
- `ALLOWED_HOSTS` makes the box refuse requests addressed to any other name.

Check it:

```bash
docker logs phone
```

A healthy start shows `web server on 0.0.0.0:8062 (https)` and `Server started at [::]:8061`. A warning that the gRPC port has no authentication is normal. If you see `PermissionError` with `load_cert_chain`, the container cannot read the key; go back to step 2.2 and check the numbers and the listing.

Then check from another machine (or the phone's browser) that the page answers:

```bash
curl -sI https://ifetch.isr.tecnico.ulisboa.pt/ | head -3
```

A normal HTTP answer means it works. A timeout means port 443 is blocked on the way to the server. A certificate error means the wrong file or hostname was used.

### 2.4 If you cannot get a certificate

You can run with the box's own throw-away certificate instead, as in section 1:

```bash
docker run -d --name phone --restart unless-stopped \
  -p 443:8062 -p 9079:8061 \
  -e TLS_SELF_SIGNED=1 \
  -e TLS_SAN=ifetch.isr.tecnico.ulisboa.pt \
  -e PUBLIC_URL=https://ifetch.isr.tecnico.ulisboa.pt \
  docker.io/sipgisr/visionist-phone:latest
```

Phones will show a certificate warning every time, and some will not give camera access on it. A tunnel service in front of the web port is an alternative that needs no certificate on the server.

### 2.5 Connect from your other computer

Because the gRPC port has no password, the safest way to reach it is an **SSH tunnel**. On the computer where your Python runs, open a terminal and leave this running:

```bash
ssh -N -L 9079:localhost:9079 your_username@ifetch.isr.tecnico.ulisboa.pt
```

While it is open, `localhost:9079` on your computer leads to the box on the server. The tunnel only works if the container was started with `-p 9079:8061` (or `-p 127.0.0.1:9079:8061`). To avoid exposing the port to the internet at all, change that option to `-p 127.0.0.1:9079:8061` when you start the container.

If you prefer not to tunnel, the alternative is to start the container with `-p 9079:8061` and allow port 9079 in the server's firewall **only from your computer's address**.

### 2.6 Create the pairing and get the token

From the computer running Python, with the tunnel open:

```python
from visionist_client import Visionist

phone = Visionist("localhost:9079")
r = phone.run(config={"phone": {
    "command": "setup",
    "title": "Annotate",
    "labels": ["handle", "chip", "ok"],
}})

info = r.config["phone"]
print(info["url"])                           # e.g. https://ifetch.isr.tecnico.ulisboa.pt/#t=...
print(info.get("warnings"))

with open("pair.png", "wb") as f:
    f.write(r.fields["qr"])                  # scan this with the phone
```

The token is the text after `#t=` in the link, exactly as in section 1.3. Open the link or scan the QR code on the phone. Because this is a real certificate, there is no warning, and the camera works with a live preview.

Calling `setup` again with the same session keeps the same link. Add `"rotate": True` to get a new token and lock out phones paired with the old one.

### 2.7 Get photos and reply

The code is exactly the same as in sections 1.6 and 1.7, because your program talks to `localhost:9079` through the tunnel. Poll with `command: get`, answer with `command: show`.

### 2.8 Upkeep and problems

- **See what is going on:** `docker logs --tail 30 phone`
- **Restart it:** `docker restart phone`. This discards the session and unread photos, so the phone must be paired again with a new `setup`.
- **Run only one phone container.** The state is in memory and is not shared between copies.
- **Certificate check:** `sudo certbot renew --dry-run` tests the renewal. After a real renewal the hook copies the new files and the box picks them up within the hour.
- **Phone says it cannot connect:** check port 443 is open to the internet, the name points at the server, and `curl -sI https://...` works from another machine.
- **Phone says the session was not found:** the container was restarted, or someone called `setup` with `rotate`. Run `setup` again and use the new link.
- **Your program cannot connect to `localhost:9079`:** check the SSH tunnel is still running and the container shows `0.0.0.0:9079->8061/tcp` in `docker ps`.
