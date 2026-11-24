# Phone box: call examples (Python and MATLAB)

Every call below names **every parameter the command accepts**, so you can copy one and delete what you do not need. The values shown are the box's defaults unless a comment says otherwise. A parameter you leave out takes its default; for `setup`, a parameter you leave out also **resets** to its default, because `setup` always states the whole page.

The two ports: **gRPC 9079** is what these calls use. The phone (or the iPad LiDAR app) uses the **web port 9080**.

Contents: [Connect](#connect) · [setup](#setup) · [get](#get) · [latest](#latest) · [show](#show) · [status and reset](#status-and-reset) · [Which parameter belongs to which command](#which-parameter-belongs-to-which-command)

---

## Connect

**Python** (`visionist_client`)

```python
from visionist_client import Visionist

phone = Visionist("ifetch.isr.tecnico.ulisboa.pt:9079")   # host:gRPC port
```

Every call is `phone.run(config={"phone": {...}}, data={...})`. Inside the `"phone"` section: `command`, `session_id` (omit for `"default"`) and `parameters`. Scalar reply values come back in `r.config["phone"]`; payloads come back as fields (`r.images`, `r.meta`, `r.depth`, `r.qr`).

**MATLAB** (`VisionIST_matlab/matlab`; Octave: see the note at the end)

```matlab
addpath("VisionIST_matlab/matlab")
cfg = vsConfig();
cfg.hosts.phone = "ifetch.isr.tecnico.ulisboa.pt:9079";   % host:gRPC port
cfg.session_id  = "demo";                                 % default session for every call
```

Every call is `vsPhone(cfg, 'command', ..., <name>, <value>, ...)`. It returns a struct: `out.reply` is the box's reply section, and payloads arrive as files (`out.images_files`, `out.depth_file`, `out.qr_file`) or decoded JSON (`out.meta`). `vsPhone` refuses an argument that does not belong to the command, so each example below lists exactly the arguments of its command.

---

## setup

Creates the session (or changes its page) and returns the pairing link and a QR code. Run it again with different values to reconfigure; phones update by themselves.

**Python**

```python
r = phone.run(config={"phone": {
    "command": "setup",
    "session_id": "demo",
    "parameters": {
        "title": "Site survey",              # heading on the page
        "labels": ["door", "valve", "crack"],# one tap-to-toggle button each
        "mode": "single",                    # "single" | "burst" (several photos per submission)
        "burst_max": 8,                      # photos per burst, 1..MAX_IMAGES
        "camera": "environment",             # "environment" (rear) | "user" (front)
        "max_side": 1600,                    # photos shrunk to this many px (0 = original)
        "jpeg_quality": 0.9,                 # 0.3 .. 1.0 for photos
        "sensors": ["location", "orientation", "motion", "lidar"],   # any subset
        "note": True,                        # show the free-text note field
        "custom_labels": False,              # let the phone user add labels
        "require_label": False,              # refuse a photo with no label
        "stream_fps": 5,                     # Stream button: frames per second, 0.2 .. 30
        "stream_max_side": 640,              # Stream button: frame size in px (0 = original)
        "stream_quality": 0.7,               # Stream button: JPEG quality 0.3 .. 1.0
        "public_url": "https://ifetch.isr.tecnico.ulisboa.pt:9080",  # what the phone opens
        "rotate": False,                     # True = new token; old phones are refused
        "qr": True,                          # also return the QR code PNG
        "qr_scale": 8,                       # pixels per QR module, 2 .. 32
    },
}})

reply = r.config["phone"]
print(reply["status"])        # "done"
print(reply["url"])           # pairing link: https://.../#t=<token>
print(reply["token"], reply["config_version"], reply["warnings"])
open("pairing_qr.png", "wb").write(r.qr)     # data.qr is the QR code PNG
```

**MATLAB**

```matlab
r = vsPhone(cfg, ...
    'command',       "setup", ...
    'session',       "demo", ...
    'title',         "Site survey", ...
    'labels',        ["door" "valve" "crack"], ...
    'mode',          "single", ...                 % "single" | "burst"
    'burst_max',     8, ...
    'camera',        "environment", ...            % "environment" | "user"
    'max_side',      1600, ...                     % 0 = original
    'jpeg_quality',  0.9, ...
    'sensors',       ["location" "orientation" "motion" "lidar"], ...
    'note',          true, ...
    'custom_labels', false, ...
    'require_label', false, ...
    'stream_fps',    5, ...
    'stream_max_side', 640, ...
    'stream_quality', 0.7, ...
    'public_url',    "https://ifetch.isr.tecnico.ulisboa.pt:9080", ...
    'rotate',        false, ...
    'qr',            true, ...
    'qr_scale',      8);

disp(r.reply.status)          % done
disp(r.reply.url)             % pairing link
imshow(imread(r.qr_file))     % the QR code, shown on screen: scan it with the phone or the iPad app
```

Reply fields (both): `status`, `url`, `url_source`, `path`, `token`, `https`, `created`, `config`, `config_version`, `web`, `warnings`. A `warnings` entry appears, for example, when the URL is plain http (the phone page then cannot show a live camera preview).

---

## get

Photos the phone **submitted** (the Send button), oldest first. Not destructive unless `pop` is true, so a retried `get` gives the same answer.

**Python**

```python
r = phone.run(config={"phone": {
    "command": "get",
    "session_id": "demo",
    "parameters": {
        "after": 0,           # only submissions with seq > after; pass the last reply's next_after
        "max_items": 8,       # most submissions per reply, 1 .. 64
        "max_mb": 32,         # soft cap on image megabytes in one reply
        "wait_s": 10,         # long poll: wait up to this long for something new (<= MAX_WAIT_S, 20)
        "latest": False,      # True = only the newest submission
        "pop": False,         # True = delete what is returned
    },
}})

reply = r.config["phone"]
print(reply["count"], reply["next_after"], reply["pending"], reply["dropped"])
meta = r.meta                 # list, one dict per submission:
                              # seq, t, client_t, device, labels, note, sensors, n_images, first_image, mime, bytes
images = r.images             # flat list of bytes (absent when count == 0)
for m in meta:
    first, n = m["first_image"], m["n_images"]
    photos = images[first:first + n]          # the photos of this submission
    open(f"photo_{m['seq']}.jpg", "wb").write(photos[0])
after = reply["next_after"]   # use it in the next call
```

**MATLAB**

```matlab
g = vsPhone(cfg, ...
    'command',   "get", ...
    'session',   "demo", ...
    'after',     0, ...                % pass g.reply.next_after next time
    'max_items', 8, ...
    'max_mb',    32, ...
    'wait_s',    10, ...               % must stay below cfg.timeout
    'latest',    false, ...
    'pop',       false);

disp([g.reply.count g.reply.next_after g.reply.pending g.reply.dropped])
if g.reply.count > 0
    files = splitlines(strtrim(string(g.images_files)));   % one path per photo
    meta  = g.meta;                                         % struct array (cell array if entries differ)
    img   = imread(files(meta(1).first_image + 1));         % first_image is 0-based
    after = g.reply.next_after;
end
```

Reply fields (both): `status`, `count`, `images` (number of photos), `seq` (newest seq in the session), `pending`, `dropped`, `next_after`. `dropped` is how many submissions aged out before you read them.

---

## latest

The **newest frame**: a frame streamed with the page's **Stream** button, a photo submitted with Send, or a frame with LiDAR depth from the iPad app (whichever came last). Frames are not queued: each one replaces the last, so a stream cannot fill the box. Read-only.

**Python**

```python
import numpy as np

r = phone.run(config={"phone": {
    "command": "latest",
    "session_id": "demo",
    "parameters": {
        "after": 0,               # return the frame only if its n > after; pass the last reply's n
        "wait_s": 5,              # long poll: wait for a newer frame (<= MAX_WAIT_S)
        "include_images": True,   # False = metadata only (cheap to poll)
    },
}})

reply = r.config["phone"]
print(reply["count"], reply["n"], reply["streaming"], reply["has_latest"])
if reply["count"] == 1:
    m = r.meta                    # ONE dict (not a list): n, source ("stream"|"photo"), seq, age_s,
                                  # t, client_t, device, labels, note, sensors, size, n_images, mime, bytes, depth
    jpeg = r.images[0]            # the frame
    if m["depth"]:                # only when the iPad LiDAR app sent depth
        d = m["depth"]            # width, height, dtype, unit, bytes, intrinsics {fx, fy, cx, cy}
        Z = np.frombuffer(r.depth, "<f4").reshape(d["height"], d["width"])   # metres
    after = reply["n"]
```

**MATLAB**

```matlab
g = vsPhone(cfg, ...
    'command',        "latest", ...
    'session',        "demo", ...
    'after',          0, ...            % pass g.reply.n next time
    'wait_s',         5, ...
    'include_images', true);

disp([g.reply.count g.reply.n g.reply.streaming g.reply.has_latest])
if g.reply.count == 1
    m   = g.meta;                                 % ONE struct: n, source, age_s, labels, sensors, size, depth ...
    img = imread(strtrim(string(g.images_files)));% the frame (one file)
    if ~isempty(m.depth)                          % only with iPad LiDAR depth
        d   = m.depth;                            % width, height, dtype, unit, intrinsics
        fid = fopen(g.depth_file, 'r');
        z   = fread(fid, inf, 'float32=>single', 0, 'l');  fclose(fid);
        Z   = reshape(z, d.width, d.height).';    % metres, height x width
    end
    after = g.reply.n;
end
```

Reply fields (both): `status`, `count` (0 or 1), `n` (rises by one per frame), `streaming` (a frame arrived in the last 3 s), `has_latest`. `age_s` in the metadata says how old the frame is. `latest` the **command** is not the same as `get`'s `latest` **flag** (which picks the newest queued submission).

---

## show

Pushes a result to the phone. On the web page the newest result appears in the fixed panel under the camera; older ones stay in the Results list when not streaming.

**Python**

```python
from pathlib import Path

r = phone.run(
    config={"phone": {
        "command": "show",
        "session_id": "demo",
        "parameters": {
            "title": "Detection",       # heading of the card (also accepted: data.text for the text)
            "text": "2 doors, 1 crack", # up to 20 000 characters
            "ref_seq": 12,              # the submission this answers: the card says "Re #12"
            "replace": True,            # clear earlier results first (use this when streaming)
        },
    }},
    data={
        "image": Path("annotated.jpg"),   # JPEG, PNG or WebP (bytes or a Path)
        "glb": Path("model.glb"),         # optional binary glTF, shown as a rotatable 3D view
        # "text": "2 doors, 1 crack",     # alternative to parameters.text
    },
)
print(r.config["phone"]["status"], r.config["phone"]["card_id"])
```

**MATLAB**

```matlab
s = vsPhone(cfg, ...
    'command', "show", ...
    'session', "demo", ...
    'title',   "Detection", ...
    'text',    "2 doors, 1 crack", ...
    'image',   "annotated.jpg", ...      % path of a JPEG/PNG/WebP file
    'glb',     "model.glb", ...          % path of a binary glTF (optional)
    'ref_seq', 12, ...
    'replace', true);
disp(s.reply.card_id)
```

`show` needs at least one of text, image, glb; otherwise the reply status is `empty_request`. Reply fields: `status`, `card_id`.

---

## status and reset

**Python**

```python
s = phone.run(config={"phone": {"command": "status", "session_id": "demo"}})
print(s.config["phone"])   # exists, sessions, queue{count,bytes,oldest_seq,newest_seq}, latest, streaming,
                           # devices[{id,name,last_seen_s}], phone_last_seen_s, program_last_poll_s, web{...}

phone.run(config={"phone": {"command": "reset", "session_id": "demo"}})   # drops that session
phone.run(config={"phone": {"command": "reset"}})                          # drops ALL sessions
```

**MATLAB**

```matlab
s = vsPhone(cfg, 'command', "status", 'session', "demo");
disp(s.reply)
vsPhone(cfg, 'command', "reset", 'session', "demo");   % drops that session
```

After a `reset`, phones paired with the session are refused; call `setup` again and pair again.

---

## Which parameter belongs to which command

| parameter | setup | get | latest | show |
|---|:-:|:-:|:-:|:-:|
| `title` | page heading | | | card heading |
| `labels`, `mode`, `burst_max`, `camera`, `max_side`, `jpeg_quality`, `sensors`, `note`, `custom_labels`, `require_label` | yes | | | |
| `stream_fps`, `stream_max_side`, `stream_quality` | yes | | | |
| `public_url`, `rotate`, `qr`, `qr_scale` | yes | | | |
| `after` | | seq cursor | frame `n` | |
| `wait_s` | | yes | yes | |
| `max_items`, `max_mb`, `latest` (flag), `pop` | | yes | | |
| `include_images` | | | yes | |
| `text`, `ref_seq`, `replace`; data `image`, `glb` | | | | yes |

`session_id` (Python) or `'session'` (MATLAB) is accepted by every command.

**Octave** (`VisionIST_matlab/octave`) takes the same names but char/cellstr values: `'labels', {'door','valve'}`, `'sensors', {'lidar'}`, `'command', 'setup'`; `g.meta` for `get` is a cell array (`g.meta{1}.seq`) and `g.images_files` is split with `strsplit(strtrim(g.images_files), "\n")`.
