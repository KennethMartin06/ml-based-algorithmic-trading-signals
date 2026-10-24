"""Verify the deck's motion three independent ways and write presentation/preview/motion_report.md.

1. Read-back: reopen the saved final deck in PowerPoint (COM) and list every slide's transition and every
   animation effect (target shape, effect type, trigger, delay, duration). Compare against motion_plan.json.
2. File check: parse the .pptx XML. Morph must appear as <p159:morph> (with PowerPoint's own fade fallback), each
   planned effect must be in the slide's <p:timing> tree, and every Morph pair must share "!!" object names.
3. Playback: export a real MP4 with PowerPoint's own renderer (Presentation.CreateVideo), extract frames with
   ffmpeg and measure them. A Morph is confirmed when the frames between two slides change gradually over
   many frames instead of in one cut, and filmstrips of those frames are saved as evidence.

A static render (render_preview.py) cannot show motion; step 3 is what demonstrates that effects actually play.

Usage:  python presentation/scripts/verify_motion.py [--skip-video]
Needs:  Windows + PowerPoint, and ffmpeg (on PATH, in $FFMPEG, or the npm ffmpeg-static package).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"
FINAL = PRES / "final_presentation.pptx"
PLAN = PRES / "build" / "motion_plan.json"
PREVIEW = PRES / "preview"
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p159": "http://schemas.microsoft.com/office/powerpoint/2015/09/main",
      "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
FPS = 10  # analysis frame rate


# ---------------------------------------------------------------------------------------------
# 1. read-back through PowerPoint
# ---------------------------------------------------------------------------------------------


def readback() -> list[dict]:
    import win32com.client as w
    from win32com.client import constants as C

    app = w.gencache.EnsureDispatch("PowerPoint.Application")
    names = {C.ppEffectMorphByObject: "Morph (objects)", C.ppEffectFadeSmoothly: "Fade (smoothly)"}
    etypes = {C.msoAnimEffectAscend: "Float In", C.msoAnimEffectFade: "Fade", C.msoAnimEffectWipe: "Wipe",
              C.msoAnimEffectFadedZoom: "Zoom"}
    trig = {C.msoAnimTriggerAfterPrevious: "after", C.msoAnimTriggerWithPrevious: "with",
            C.msoAnimTriggerOnPageClick: "click"}
    pres = app.Presentations.Open(str(FINAL), True, False, False)
    out = []
    try:
        for s in pres.Slides:
            t = s.SlideShowTransition
            seq = s.TimeLine.MainSequence
            effs = []
            for i in range(1, seq.Count + 1):
                e = seq.Item(i)
                effs.append({"shape": e.Shape.Name, "effect": etypes.get(e.EffectType, str(e.EffectType)),
                             "trigger": trig.get(e.Timing.TriggerType, str(e.Timing.TriggerType)),
                             "delay": round(e.Timing.TriggerDelayTime, 3), "dur": round(e.Timing.Duration, 3),
                             "exit": bool(e.Exit)})
            out.append({"slide": s.SlideIndex, "transition": names.get(t.EntryEffect, str(t.EntryEffect)),
                        "transition_dur": round(t.Duration, 3), "effects": effs})
    finally:
        pres.Close()
    return out


# ---------------------------------------------------------------------------------------------
# 2. file-level checks
# ---------------------------------------------------------------------------------------------


def file_checks(plan: list[dict]) -> tuple[list[dict], list[str]]:
    z = zipfile.ZipFile(FINAL)
    pres = etree.fromstring(z.read("ppt/presentation.xml"))
    rels = etree.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    order = [target[s.get(f"{{{NS['r']}}}id")] for s in pres.find("p:sldIdLst", NS)]
    rows, names_by_slide = [], []
    for i, part in enumerate(order, 1):
        x = etree.fromstring(z.read("ppt/" + part))
        ids = {e.get("id"): e.get("name") for e in x.iter(f"{{{NS['p']}}}cNvPr")}
        names_by_slide.append(set(ids.values()))
        morph = x.find(".//mc:AlternateContent/mc:Choice/p:transition/p159:morph", NS)
        fallback = x.find(".//mc:AlternateContent/mc:Fallback/p:transition", NS)
        fade = x.find(".//p:transition/p:fade", NS)
        pars = x.findall(".//p:timing//p:cTn[@presetClass='entr']", NS)
        targets = [ids.get(c.find(".//p:spTgt", NS).get("spid")) for c in pars]
        presets = sorted({(c.get("presetID"), c.get("presetSubtype")) for c in pars})
        rows.append({"slide": i, "xml_transition": "morph" if morph is not None else ("fade" if fade is not None
                                                                                     else "none"),
                     "morph_option": morph.get("option") if morph is not None else None,
                     "has_fallback": fallback is not None, "entrance_effects": len(pars),
                     "targets": targets, "presets": presets})
    problems = []
    for row, spec in zip(rows, plan):
        planned = [s["shape"] for s in spec["steps"]]
        if row["xml_transition"] != spec["transition"]:
            problems.append(f"slide {row['slide']}: transition {row['xml_transition']} != plan {spec['transition']}")
        if sorted(t or "" for t in row["targets"]) != sorted(planned):
            problems.append(f"slide {row['slide']}: animated shapes differ from the plan")
        if spec["transition"] == "morph":
            prev, cur = names_by_slide[row["slide"] - 2], names_by_slide[row["slide"] - 1]
            shared = sorted(n for n in cur & prev if n and n.startswith("!!"))
            row["morph_shared_objects"] = shared
            if not shared:
                problems.append(f"slide {row['slide']}: Morph without shared '!!' objects")
    return rows, problems


# ---------------------------------------------------------------------------------------------
# 3. playback through a PowerPoint video export
# ---------------------------------------------------------------------------------------------


def ffmpeg_bin() -> str:
    cands = [os.environ.get("FFMPEG"), shutil.which("ffmpeg"),
             str(Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "ffmpeg-static" / "ffmpeg.exe")]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("ffmpeg not found; set FFMPEG or install ffmpeg")


def export_video(mp4: Path, slide_seconds: float) -> float:
    import win32com.client as w
    from win32com.client import constants as C

    app = w.gencache.EnsureDispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(FINAL), True, False, False)
    t0 = time.time()
    try:
        pres.CreateVideo(str(mp4), False, slide_seconds, 720, 30, 85)
        while pres.CreateVideoStatus in (C.ppMediaTaskStatusInProgress, C.ppMediaTaskStatusQueued):
            time.sleep(2)
        if pres.CreateVideoStatus != C.ppMediaTaskStatusDone:
            raise RuntimeError(f"CreateVideo failed with status {pres.CreateVideoStatus}")
    finally:
        pres.Close()
    return time.time() - t0


def load_small(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((192, 108), Image.BILINEAR), dtype=np.float32)


def analyse_video(mp4: Path, statics: list[Path], plan: list[dict], workdir: Path) -> dict:
    ff = ffmpeg_bin()
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-loglevel", "error", "-i", str(mp4), "-vf", f"fps={FPS},scale=384:216",
                    str(frames_dir / "f%05d.png")], check=True)
    frames = sorted(frames_dir.glob("f*.png"))
    F = np.stack([load_small(p) for p in frames])
    S = np.stack([load_small(p) for p in statics])
    mse = ((F[:, None] - S[None]) ** 2).mean(axis=(2, 3))  # frames x slides
    label = mse.argmin(axis=1) + 1
    diff = np.r_[0.0, np.abs(np.diff(F, axis=0)).mean(axis=(1, 2))]

    # slide k "starts" at the first frame whose nearest static render is k (monotone in time)
    starts, pos = {}, 0
    for k in range(1, len(statics) + 1):
        idx = np.nonzero(label[pos:] == k)[0]
        if len(idx) == 0:
            continue
        starts[k] = pos + int(idx[0])
        pos = starts[k]

    results = {"frames": len(frames), "fps": FPS, "duration_s": round(len(frames) / FPS, 1),
               "slides_seen": sorted(starts), "transitions": []}
    for k, spec in enumerate(plan, 1):
        if k == 1 or k not in starts:
            continue
        i = starts[k]
        lo, hi = max(0, i - int(2.5 * FPS)), min(len(frames) - 1, i + int(2.5 * FPS))
        # the contiguous run of moving frames around the switch point
        run = []
        for j in range(i, lo - 1, -1):
            if diff[j] > 0.35:
                run.insert(0, j)
            elif run:
                break
        for j in range(i + 1, hi + 1):
            if diff[j] > 0.35:
                run.append(j)
            else:
                break
        results["transitions"].append({"into_slide": k, "planned": spec["transition"],
                                       "planned_s": spec["duration"], "switch_frame": i,
                                       "moving_frames": len(run), "moving_s": round(len(run) / FPS, 2),
                                       "max_frame_diff": round(float(diff[lo:hi + 1].max()), 2),
                                       "run": [run[0], run[-1]] if run else None})
    results["_label"], results["_diff"] = label.tolist(), diff.round(3).tolist()
    return results


def filmstrip(mp4: Path, t_start: float, t_end: float, n: int, out: Path, title: str) -> None:
    ff = ffmpeg_bin()
    tmp = Path(tempfile.mkdtemp())
    times = np.linspace(t_start, t_end, n)
    ims = []
    for j, t in enumerate(times):
        f = tmp / f"s{j}.png"
        subprocess.run([ff, "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1", "-vf",
                        "scale=480:270", str(f)], check=True)
        ims.append(Image.open(f).convert("RGB"))
    pad, head = 10, 34
    W = n * 480 + (n + 1) * pad
    sheet = Image.new("RGB", (W, 270 + head + 2 * pad + 22), (18, 22, 28))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("consola.ttf", 18)
        small = ImageFont.truetype("consola.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((pad, 8), title, fill=(255, 181, 71), font=font)
    for j, (im, t) in enumerate(zip(ims, times)):
        x = pad + j * (480 + pad)
        sheet.paste(im, (x, head + pad))
        d.text((x, head + pad + 274), f"t = {t:6.2f} s", fill=(170, 182, 195), font=small)
    sheet.save(out)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--slide-seconds", type=float, default=2.0)
    args = ap.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))["slides"]
    PREVIEW.mkdir(parents=True, exist_ok=True)
    motion_dir = PREVIEW / "motion"
    motion_dir.mkdir(exist_ok=True)

    rb = readback()
    rows, problems = file_checks(plan)
    for spec, r in zip(plan, rb):
        if len(r["effects"]) != len(spec["steps"]):
            problems.append(f"slide {r['slide']}: PowerPoint reports {len(r['effects'])} effects, plan has "
                            f"{len(spec['steps'])}")
        want = "Morph (objects)" if spec["transition"] == "morph" else "Fade (smoothly)"
        if r["transition"] != want:
            problems.append(f"slide {r['slide']}: PowerPoint reports {r['transition']}, plan {want}")

    video = None
    if not args.skip_video:
        statics = sorted(PREVIEW.glob("slide-*.png"))
        assert len(statics) == len(plan), "run render_preview.py on the final deck first"
        work = Path(tempfile.mkdtemp())
        mp4 = work / "deck.mp4"
        secs = export_video(mp4, args.slide_seconds)
        video = analyse_video(mp4, statics, plan, work)
        video["export_seconds"] = round(secs, 1)
        for tr in video["transitions"]:
            if tr["planned"] == "morph" and tr["run"]:
                a, b = tr["run"]
                filmstrip(mp4, (a - 1) / FPS, (b + 1) / FPS, 6, motion_dir / f"morph_into_slide_{tr['into_slide']:02d}.png",
                          f"Morph into slide {tr['into_slide']}: frames from PowerPoint's own video export")
        s7 = next((t for t in video["transitions"] if t["into_slide"] == 7), None)
        s8 = next((t for t in video["transitions"] if t["into_slide"] == 8), None)
        if s7 and s8:
            filmstrip(mp4, s7["switch_frame"] / FPS, s8["switch_frame"] / FPS - 0.2, 6,
                      motion_dir / "build_slide_07.png", "Slide 7 progressive build (architecture)")
        s2 = next((t for t in video["transitions"] if t["into_slide"] == 2), None)
        if s2:
            filmstrip(mp4, 0.0, s2["switch_frame"] / FPS - 0.2, 6, motion_dir / "build_slide_01.png",
                      "Slide 1 entrance sequence (title, price trace drawn left to right)")
        keep = PREVIEW / "motion" / "deck_preview.mp4"
        shutil.copy(mp4, keep)
        video["mp4"] = str(keep.relative_to(ROOT))
        for tr in video["transitions"]:
            if tr["planned"] == "morph" and tr["moving_frames"] < 5:
                problems.append(f"slide {tr['into_slide']}: Morph did not animate gradually in the video "
                                f"({tr['moving_frames']} moving frames)")
        shutil.rmtree(work, ignore_errors=True)

    report = {"deck": FINAL.name, "readback": rb, "file": rows, "video": video, "problems": problems}
    (motion_dir / "motion_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, plan, PREVIEW / "motion_report.md")
    print("problems:", problems or "none")


def write_markdown(rep: dict, plan: list[dict], path: Path) -> None:
    L = ["# Motion verification report", "",
         f"Deck: `presentation/{rep['deck']}` · generated by `presentation/scripts/verify_motion.py`", "",
         "## 1. Read back from PowerPoint (COM) and the file XML", "",
         "| Slide | Transition (PowerPoint) | XML | Effects (PowerPoint / plan) | Clicks | Morph objects shared with previous slide |",
         "|---:|---|---|---:|---:|---|"]
    for r, f, spec in zip(rep["readback"], rep["file"], plan):
        clicks = sum(1 for e in r["effects"] if e["trigger"] == "click")
        xml = f["xml_transition"] + (f" ({f['morph_option']}, fade fallback)" if f["morph_option"] else "")
        shared = ", ".join(f.get("morph_shared_objects", [])) or "—"
        L.append(f"| {r['slide']} | {r['transition']} · {r['transition_dur']} s | {xml} | {len(r['effects'])} / "
                 f"{len(spec['steps'])} | {clicks} | {shared} |")
    L += ["", "Effect types used (PowerPoint presetID / subtype): " +
          ", ".join(sorted({f"{p[0]}/{p[1]}" for f in rep["file"] for p in f["presets"]})), ""]
    v = rep["video"]
    if v:
        L += ["## 2. Playback in PowerPoint's own video export", "",
              f"PowerPoint rendered the deck to MP4 (`{v['mp4']}`, {v['duration_s']} s, exported in "
              f"{v['export_seconds']} s). Frames were sampled at {v['fps']} fps and matched against the static "
              "renders. A *moving frame* is one that differs visibly from the previous frame; a cut produces one, "
              "an animated transition produces many.", "",
              "| Into slide | Planned | Planned duration | Moving frames at the switch | ≈ seconds of motion |",
              "|---:|---|---:|---:|---:|"]
        for t in v["transitions"]:
            L.append(f"| {t['into_slide']} | {t['planned']} | {t['planned_s']} s | {t['moving_frames']} | "
                     f"{t['moving_s']} |")
        L += ["", f"Slides seen in the video: {len(v['slides_seen'])} of {len(plan)}.", "",
              "Filmstrips in `preview/motion/` show the in-between frames of each Morph and two entrance "
              "sequences.", ""]
    L += ["## Problems", "", *(f"- {p}" for p in rep["problems"])] if rep["problems"] else ["## Problems", "",
                                                                                       "None found."]
    L += ["", "## What this does and does not prove", "",
          "- The read-back and XML checks prove which transitions and effects are configured, on which objects, "
          "in what order and with what timing.",
          "- The video export proves that PowerPoint itself plays them: Morph slides change over many frames "
          "rather than cutting.",
          "- Neither replaces a rehearsal on the presentation machine. Morph needs PowerPoint 2019, 2021 or "
          "Microsoft 365. Older versions and Google Slides show the fade fallback instead.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
