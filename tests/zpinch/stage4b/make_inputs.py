#!/usr/bin/env python3
"""Stage 4b test inputs from the production templates gen/A2T1, gen/A3aT2 (campaign.py make, production cadence).
  verify/V1_spec_T1   A2-T1 turbulent state (realtest t0500, reset to t = 0): 0d energy+invariants, 1d zp_spectra,
                      maps zp_maps+zp_kxky (n = 1 at t = 0), kpar; save 0.1, tfinal 0.2
  verify/V2_lx8pi     tiny 48x24x48, L_x = 8 pi, white noise 1e-3, tfinal 1: 1d/maps/kpar save 0.5; final state kept
  verify/V3_signtest  A2-T1 state, kP = 0, nu = 0, dt = 1e-4, 8 steps: 0d energy and zp_shells EVERY STEP
  overhead/O1_T1, O2_T2   production cadence (all outputs) from turbulent states (A2-T1 t500, A3a-T2 t1200), 10 R/vA
"""
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
ST_T1 = "../../../stage4a/realtest/A2-T1-k8/t0500/state.nc"
ST_T2 = "../../../stage3e/A3a-T2-k8/t1200/restart.nc"


def tmpl(name):
    return open(os.path.join(H, "gen", name, "t0100", "input.toml")).read()


def outputs_only(txt, keep):
    """drop [output.X] blocks not in keep"""
    blocks = re.split(r"(?m)^(?=\[)", txt)
    out = []
    for b in blocks:
        m = re.match(r"\[output\.(\w+)\]", b)
        if m and m.group(1) not in keep:
            continue
        out.append(b)
    return "".join(out)


def sub(txt, pat, rep):
    new, n = re.subn(pat, rep, txt, flags=re.M)
    assert n == 1, (pat, n)
    return new


def restart(txt, src, write=False):
    txt = sub(txt, r"^\[restart\][\s\S]*\Z", f'[restart]\nrestart_from = "{src}"\nreset_time = true\n'
              f'write_restart_file = {"true" if write else "false"}\n')
    return txt


def write(path, txt):
    os.makedirs(os.path.join(H, path), exist_ok=True)
    open(os.path.join(H, path, "input.toml"), "w").write(txt)
    print("wrote", path)


# V1
t = outputs_only(tmpl("A2T1"), {"0d", "1d", "maps", "kpar"})
t = sub(t, r"^tfinal = .*$", "tfinal = 0.2")
t = sub(t, r'^diags = \[\{name = "zp_spectra"\}, \{name = "zp_profiles"\}\]', 'diags = [{name = "zp_spectra"}]')
for o in ("1d", "maps", "kpar"):
    t = sub(t, rf"(\[output\.{o}\]\n)save_time = .*", r"\1save_time = 0.1")
t = sub(t, r"(\[output\.maps\]\nsave_time = 0.1\n)sample_time = .*", r"\1sample_time = 0.05")
write("verify/V1_spec_T1", restart(t, ST_T1))

# V2
t = outputs_only(tmpl("A2T1"), {"0d", "1d", "maps", "kpar"})
t = sub(t, r"^nx = .*$", "nx = 48")
t = sub(t, r"^ny = .*$", "ny = 24")
t = sub(t, r"^nz = .*$", "nz = 48\nLx = 25.132741228718345  # 8 pi")
t = sub(t, r"^kperp = .*$", "kperp = 2e-06")
t = sub(t, r"^kz = .*$", "kz = 1e-08")
t = sub(t, r"^amplitude = .*$", "amplitude = 0.001")
t = sub(t, r"^tfinal = .*$", "tfinal = 1.0")
t = sub(t, r'^diags = \[\{name = "zp_spectra"\}, \{name = "zp_profiles"\}\]', 'diags = [{name = "zp_spectra"}]')
for o in ("1d", "maps", "kpar"):
    t = sub(t, rf"(\[output\.{o}\]\n)save_time = .*", r"\1save_time = 0.5")
t = sub(t, r"(\[output\.maps\]\nsave_time = 0.5\n)sample_time = .*", r"\1sample_time = 0.25")
write("verify/V2_lx8pi", t)

# V3
t = outputs_only(tmpl("A2T1"), {"0d", "shells"})
t = sub(t, r"^kP = .*$", "kP = 0.0")
t = sub(t, r"^kperp = .*$", "kperp = 0.0")
t = sub(t, r"^kz = .*$", "kz = 0.0")
t = sub(t, r"^dt_max = .*$", "dt_max = 0.0001")
t = sub(t, r"^tfinal = .*$", "tfinal = 0.00080001")
t = sub(t, r'^diags = \[\{name = "energy".*$', 'diags = [{name = "energy", options = {invariants = true}}]')
t = sub(t, r"(\[output\.0d\]\n)save_time = .*", r"\1save_steps = 1")
t = sub(t, r"(\[output\.shells\]\n)save_time = .*", r"\1save_steps = 1")
write("verify/V3_signtest", restart(t, ST_T1))

# O1, O2
for name, tm, st in (("overhead/O1_T1", "A2T1", ST_T1), ("overhead/O2_T2", "A3aT2", ST_T2)):
    t = sub(tmpl(tm), r"^tfinal = .*$", "tfinal = 10.0")
    write(name, restart(t, st))
