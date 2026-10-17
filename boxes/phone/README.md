# phone

A phone as a capture and annotation terminal, **and the phone side is a box like any other**. The person on the phone opens a web page, frames an object, takes a photo, taps some labels and sends. Nothing is pushed to your program: the photos wait inside the box, and your program fetches them with ordinary `run()` calls and keeps polling for more. Results go back the same way with `show`.

```python
from visionist_client import Visionist

phone = Visionist("phonebox:9079")                       # the gRPC port, as always
setup = phone.run(config={"phone": {"command": "setup",
                                    "title": "Mug check",
                                    "labels": ["handle", "chip", "ok"],
                                    "public_url": "https://lab.example.org/phone"}})
# The reply carries `url` (the pairing link) and data.qr (a PNG): show it, scan it.

after = 0
while True:
    r = phone.run(config={"phone": {"command": "get", "after": after, "wait_s": 15}})
    # reply config: count, next_after, dropped ...
    # data.meta   : JSON list, one entry per submission:
    #               seq, t, labels, note, sensors, device, n_images, first_image, mime, bytes
    # data.images : the photos of this reply, flat; entry i owns
    #               images[first_image : first_image + n_images]
    after = ...                                          # the reply's next_after
```

The exact accessors for reply fields depend on the client version (see the repo's client docs); the wire shapes above are what this box guarantees.

Two ports are involved, and they are not alike:

| port | host default | who talks to it | protocol |
|---|---|---|---|
| gRPC `8061` | `9079` | **your programs** | Visionist `Process`, no authentication |
| web `8062` | `9080` | **phones** | HTTP, or HTTPS with `TLS_CERT`/`TLS_KEY` |

The web port is declared as an `extra_ports` entry in `box.yaml` (`name: web`, env `WEB_PORT`), which is how the registry, the port ledger and the fleet generator know about it.

## Commands

Config section `phone`; `session_id` at the top of the section (omitted = `"default"`). A session pairs one program with its phones: own page configuration, token, capture queue and results feed.

| command | what it does |
|---|---|
| `setup` | Create or reconfigure the session; returns the pairing `url` (and `path`), `token`, a QR PNG (`data.qr`) and `warnings`. Page parameters: `title`, `labels`, `mode` (`single`/`burst`), `burst_max`, `camera`, `max_side`, `jpeg_quality`, `sensors` (`location`, `orientation`, `motion`, `lidar`), `note`, `custom_labels`, `require_label`, and for streaming `stream_fps` (default 5), `stream_max_side` (640), `stream_quality` (0.7). `rotate: true` issues a new token and locks out phones paired with the old one. |
| `get` | Submissions with `seq > after`. Replies carry `count`, `next_after`, `dropped` (how many aged out or were evicted before you read them) and `data.images` plus `data.meta` (JSON, one entry per submission). `wait_s` long-polls, `latest` returns only the newest, `pop` deletes what it returns, `max_items`/`max_mb` bound a reply. Not destructive by default, so a retried `get` gives the same answer. |
| `latest` | **Real time.** The newest frame (streamed with the page's *Stream* button) or submitted photo, with its labels, note, sensors and optional depth. Read-only and never consumes anything. Reply carries `count` (0 or 1), `n` (a counter that only increases), `streaming` (a frame arrived in the last 3 s) and `has_latest`; `data.images` (one JPEG/PNG/WebP), `data.meta` (**one JSON object**, not a list: `n`, `source` = `stream` or `photo`, `seq` (photos only), `age_s`, `labels`, `note`, `sensors` (a one-element list), `size`, `device`, `depth`) and `data.depth` when there is one. `after` = the `n` you last handled returns `count: 0` if nothing is newer; `wait_s` long-polls; `include_images: false` returns metadata only. (Not to be confused with `get`'s `latest: true` flag, which picks the newest of the *queued submissions*.) |
| `show` | Push a result to the phone's results feed: `text` (or `data.text`), `data.image`, `data.glb` (a 3D model, shown in an orbit viewer), `title`, `ref_seq` ("Re #12"), `replace`. The page shows the newest result in a **fixed live panel** under the camera (text, title and image replaced together, never moving the layout); the *Results* list below keeps the history when not streaming. |
| `status` | Whether the session exists, all session ids, queue state, and the web settings in force. |
| `reset` | Remove the session (queue, feed and token); reports `sessions_removed`. Phones paired with it are refused afterwards; `setup` again to re-pair. |

Errors use the usual `status: error`; `get` never blocks longer than `MAX_WAIT_S`, so keep your gRPC timeout above it.

## Streaming

The page has a **Stream** button over the live camera. On, it sends frames to the box (default 5 fps, longest side 640 px, JPEG 0.7; `setup` changes all three) and the newest result you `show` appears in the fixed panel under the camera, in real time. Off, it stops. Photos, labels, note and sensors work as before, and while streaming each frame carries the current labels, note and sensor readings.

Frames are **not queued**. Each one overwrites a single `latest` slot per session (a submitted photo overwrites it too), so a stream can run for hours without filling the box and `get` is untouched. Frames go out one at a time: a slow link lowers the frame rate rather than building a backlog. Streaming needs the live camera, so like the camera it needs https or localhost; over plain http the button is hidden.

```python
phone.run(config={"phone": {"command": "setup", "labels": ["person", "car"],
                            "sensors": ["orientation", "lidar"], "stream_fps": 10,
                            "public_url": "https://lab.example.org/phone"}})
n = 0
while True:
    r = phone.run(config={"phone": {"command": "latest", "after": n, "wait_s": 5}})
    # reply: count, n, streaming; data.images[0] = the JPEG, data.meta = one object
    if r.count == 0:
        continue                       # nothing newer than frame n
    n = r.n
    jpeg, meta = r.images[0], r.meta   # meta["labels"], meta["sensors"], meta["age_s"] ...
    phone.run(config={"phone": {"command": "show", "text": "3 people", "replace": True}},
              data={"image": annotated_jpeg})        # lands in the fixed panel
```

Show with `replace: true` when streaming: results arrive as fast as you send them and the phone only keeps the newest in the live panel.

### LiDAR (iPhone / iPad Pro) and depth

**A web page cannot read LiDAR.** iOS Safari has no WebXR and ARKit's depth is native-only. So with `sensors` including `lidar` the page honestly reports `sensors[0].lidar = {"available": 0, "webxr": 0|1}` in every frame, so your program can tell "no LiDAR" from "forgot to ask".

The box side is complete, so a small **native client** (ARKit `sceneDepth` / `smoothedSceneDepth`) can supply real depth. It talks to the same web port with the pairing token:

```
POST {PUBLIC_URL}/api/frame            Authorization: Bearer <token>
multipart/form-data:
  image  JPEG/PNG/WebP (the camera frame)
  depth  raw bytes, row-major, little-endian, no header
  meta   JSON: {"depth": {"width": 256, "height": 192, "dtype": "float32", "unit": "m",
                          "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}},
                "sensors": [{"lidar": {"available": 1, "min_m": 0.3, "max_m": 4.9}}],
                "labels": [], "size": [w, h], "device": {"id": "...", "name": "iPad"}}
```

`dtype` is `float32` or `uint16`, `unit` is `m` or `mm`; the box checks that the byte count equals `width * height * itemsize` and refuses anything else. `latest` then returns it untouched as `data.depth` with `meta.depth` describing it: `np.frombuffer(depth, "<f4").reshape(h, w)`. The `lidar` sensor group is free-form numbers (a summary such as min/max/centre distance is handy); the box never decodes depth or images.

## Pairing and security

`setup` returns a link of the form `https://host/#t=<token>`. The token is in the URL **fragment**, so it never reaches proxy or server logs; the page reads it and sends `Authorization: Bearer`. Anyone holding the link can submit photos and see results to that session, so treat it like a password and `rotate` it when it leaks.

Built in: per-session token, wrong-token lockout (429 after `AUTH_FAIL_LIMIT` per address per 5 minutes), per-address rate limit, size limits, image magic-number checks, `Host` allowlist, strict CSP (`script-src 'self'`, nothing loaded from third parties; the 3D viewer is bundled into the image), HSTS when served over HTTPS.

**The gRPC port has no authentication** (like every box). Anyone who can reach it can read all photos and push results. Publish it to a private network only; publish just the web port to the world.

## Camera needs HTTPS

Browsers expose the camera (`getUserMedia`) and the motion sensors only in a *secure context*: `https://` or `localhost`. Over plain HTTP the page falls back to the phone's own camera app (`<input capture>`), which works but gives no live viewfinder and no sensors. For real use serve HTTPS in one of the two ways below. `setup` warns when the pairing URL is plain HTTP.

## Deployment A: behind a reverse proxy

The proxy terminates TLS; the box speaks plain HTTP to it.

```yaml
environment:
  PUBLIC_URL: https://lab.example.org/phone    # what phones are sent to (prefix included)
  TRUST_PROXY: "1"                             # believe X-Forwarded-For from one hop
  ALLOWED_HOSTS: lab.example.org
  # BASE_PATH: /phone                          # ONLY if the proxy does not strip the prefix
```

nginx, stripping the prefix (so `BASE_PATH` stays unset):

```nginx
location /phone/ {
    proxy_pass         http://127.0.0.1:9080/;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Forwarded-For   $remote_addr;
    proxy_set_header   X-Forwarded-Proto $scheme;
    client_max_body_size 32m;
    proxy_read_timeout 60s;                    # above MAX_WAIT_S (20)
    proxy_request_buffering off;
}
```

Caddy:

```
lab.example.org {
    handle_path /phone/* {
        reverse_proxy 127.0.0.1:9080
    }
}
```

If the proxy forwards the prefix untouched, set `BASE_PATH=/phone` and drop the stripping. With `TRUST_PROXY=0` (the default) the box uses the socket address and ignores `X-Forwarded-*`, so rate limits and lockouts would all hit the proxy's address; set the right hop count. Bind the box's host ports to loopback so only the proxy reaches them: `make_fleet.py --bind 127.0.0.1` does that for gRPC ports, and the web port mapping can be edited to `127.0.0.1:9080:8062`.

## Deployment B: directly on a public server

The box terminates TLS itself.

```yaml
services:
  phone:
    environment:
      TLS_CERT: /certs/fullchain.pem
      TLS_KEY: /certs/privkey.pem
      PUBLIC_URL: https://phone.example.org:9080
      ALLOWED_HOSTS: phone.example.org
    volumes:
      - /etc/letsencrypt/live/phone.example.org:/certs:ro
    ports:
      - "127.0.0.1:9079:8061"      # gRPC: private
      - "9080:8062"                # web: public
```

The certificate files are re-read hourly, so renewals are picked up without a restart (mount the directory, not single files). For a LAN or a quick test, `TLS_SELF_SIGNED=1` (optionally `TLS_SAN=phone.lan,192.168.1.20`) generates a certificate at startup; phones warn about it and some will not give camera access over it. Port 443 needs either `WEB_PORT=443` with the right capability, or a host mapping `443:8062`.

## Environment variables

| variable | default | |
|---|---|---|
| `WEB_PORT` | 8062 | in-container web port |
| `WEB_HOST` | 0.0.0.0 | bind address |
| `PUBLIC_URL` | – | external URL of the page root; `setup` also accepts it as a parameter |
| `BASE_PATH` | – | serve under this prefix (only if the proxy keeps it) |
| `TRUST_PROXY` | 0 | proxy hops whose `X-Forwarded-For` is believed |
| `ALLOWED_HOSTS` | any | comma list; other `Host` headers get 421 |
| `TLS_CERT`, `TLS_KEY` | – | PEM files; both set → HTTPS |
| `TLS_SELF_SIGNED`, `TLS_SAN`, `TLS_DIR` | 0, auto, /tmp/phone-tls | generated certificate |
| `MAX_SUBMIT_MB` | 24 | bytes per submission (also capped to 80 % of the gRPC message limit) |
| `MAX_IMAGES` | 12 | images per submission |
| `MAX_FRAME_MB` | 8 | bytes per streamed frame (image + depth; also capped to 80 % of the gRPC message limit) |
| `FRAME_RATE_PER_S` | 30 | streamed frames per address per second (429 beyond; phones behind one NAT share it) |
| `MAX_QUEUE` | 200 | submissions kept per session |
| `MAX_STORE_MB` | 256 | all held data, all sessions |
| `MAX_SESSIONS` | 1000 | Most sessions held at once; a new `setup` beyond this is refused. |
| `SESSION_TTL_S` | 604800 | A session idle this long (one week) is removed with everything it holds (latest image, depth, queue, token). `0` = never expire. A program's `get`/`latest`/`show`/`setup` and any phone traffic count as activity; `status` does not. |
| `TTL_S` | 3600 | unread submissions and results expire |
| `RATE_PER_MIN` | 60 | submissions per address per minute |
| `AUTH_FAIL_LIMIT` | 10 | wrong tokens per address per 5 min |
| `MAX_WAIT_S` | 20 | longest long poll |
| `FEED_KEEP` | 50 | results kept for the phone |

Plus the usual `PORT`, `MAX_MESSAGE_MB`, `MAX_WORKERS` of every box.

## Things to know

- **Everything is in RAM.** Restarting the box drops queues, feeds and sessions (phones get a "session not found" and must re-pair). `dropped` in a `get` reply tells you when you were too slow.
- **One replica only.** State is in-process; do not put several phone boxes behind a load balancer.
- **Long polls and proxies.** Keep `wait_s` below `MAX_WAIT_S`, your gRPC deadline, and any proxy read timeout.
- **iOS** needs a tap before it gives orientation/motion data; the page offers the *Enable sensors* button when `sensors` asks for it.
- **Streaming and proxies.** Frames are ordinary POSTs; behind a proxy keep `client_max_body_size` above `MAX_FRAME_MB`. The page keeps the screen awake while the camera runs, but a phone that locks or backgrounds the tab stops streaming (`streaming: false` in `latest`).
- A submission must fit in one `get` reply; the size limit above guarantees that.

## Testing

```bash
python3 test/test_phone.py          # real gRPC + HTTP servers in-process, no docker or browser
```

The page itself (camera, labels, 3D viewer, CSP cleanliness) was checked in headless Chromium with a fake camera; that script is not part of the unit tests because it needs Playwright.

## Mock

Nothing to mock: the box makes no outbound connections.
