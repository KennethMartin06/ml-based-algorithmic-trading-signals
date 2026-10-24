"""Apply transitions, animations and sections to the static deck through PowerPoint (COM), then save the final deck.

Input:   presentation/build/deck_static.pptx and presentation/build/motion_plan.json (from build_deck.py)
Output:  presentation/final_presentation.pptx (fonts embedded)

PowerPoint writes the animation timing tree and the Morph transition itself, so the result is exactly what
PowerPoint would produce if the effects were added by hand in the Animations / Transitions tabs. Afterwards the
document properties are reset with python-pptx, because Office stamps "last modified by" with the local user name.

Requires Windows with Microsoft PowerPoint 2019 / Microsoft 365 (Morph needs PowerPoint 2019 or later).
Run from the project root:  python presentation/scripts/apply_motion.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"
STATIC = PRES / "build" / "deck_static.pptx"
PLAN = PRES / "build" / "motion_plan.json"
FINAL = PRES / "final_presentation.pptx"

SECTIONS = [(1, "Introduction"), (7, "System"), (9, "Method and implementation"), (14, "Key features and live demo"),
            (17, "Results"), (20, "Wrap-up")]


def main() -> None:
    import win32com.client as w
    from win32com.client import constants as C

    plan = json.loads(PLAN.read_text(encoding="utf-8"))["slides"]
    app = w.gencache.EnsureDispatch("PowerPoint.Application")  # also loads the typelib constants
    effects = {"rise": C.msoAnimEffectAscend, "fade": C.msoAnimEffectFade, "wipe": C.msoAnimEffectWipe,
               "zoom": C.msoAnimEffectFadedZoom}
    triggers = {"after": C.msoAnimTriggerAfterPrevious, "with": C.msoAnimTriggerWithPrevious,
                "click": C.msoAnimTriggerOnPageClick}
    transitions = {"fade": C.ppEffectFadeSmoothly, "morph": C.ppEffectMorphByObject}
    pres = app.Presentations.Open(str(STATIC), False, False, False)
    try:
        assert pres.Slides.Count == len(plan), (pres.Slides.Count, len(plan))
        for idx, spec in enumerate(plan, 1):
            slide = pres.Slides(idx)
            tr = slide.SlideShowTransition
            tr.EntryEffect = transitions[spec["transition"]]
            tr.Duration = spec["duration"]
            tr.AdvanceOnClick = True
            tr.AdvanceOnTime = False
            seq = slide.TimeLine.MainSequence
            while seq.Count:
                seq.Item(1).Delete()
            for step in spec["steps"]:
                shp = slide.Shapes(step["shape"])
                try:
                    eff = seq.AddEffect(shp, effects[step["effect"]], C.msoAnimateLevelNone, triggers[step["trigger"]])
                except Exception as exc:  # name the step, PowerPoint's own message is generic
                    raise RuntimeError(f"slide {idx}: cannot animate {step}") from exc
                if step.get("direction") == "left":
                    eff.EffectParameters.Direction = C.msoAnimDirectionLeft
                eff.Timing.Duration = step["dur"]
                eff.Timing.TriggerDelayTime = step["delay"]
        sp = pres.SectionProperties
        while sp.Count:
            sp.Delete(1, False)
        for first, name in SECTIONS:
            sp.AddBeforeSlide(first, name)
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        # ppSaveAsOpenXMLPresentation, embedding the fonts (Segoe UI, Consolas, Cambria are all embeddable)
        pres.SaveAs(str(FINAL), 24, -1)
    finally:
        pres.Close()

    from pptx import Presentation

    prs = Presentation(str(FINAL))
    cp = prs.core_properties
    cp.title = "Machine-Learning Trading Signals: A Comparative Study Against Technical Baseline Strategies"
    cp.subject = "B.Tech CSE (AI & ML) mini-project presentation, Manipal Institute of Technology"
    cp.author = "Ishan Singh; Kenneth Martin"
    cp.last_modified_by = "Ishan Singh; Kenneth Martin"
    cp.keywords = "machine learning; algorithmic trading; backtesting; XGBoost; LSTM"
    cp.comments = "Built by presentation/scripts (build_deck.py, apply_motion.py)."
    prs.save(str(FINAL))
    print(f"wrote {FINAL}")


if __name__ == "__main__":
    main()
