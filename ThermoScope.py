#!/usr/bin/env python3
"""Single-file packaging of the user-supplied CubicLab project.
Run: python3 CubicLab_Friend.py, then open http://127.0.0.1:8770
Requires Python 3.10+; standard library only; Ctrl+C stops the server.
The engine, HTML, CSS, JavaScript and fluid data were supplied by the user.
The HTTP adapter and executable validation checks were added for this package.
Shomate A-E data: NIST WebBook / Chase (1998); critical constants: supplied
CoolProp-derived data. Sources remain in the FLUIDS metadata.
"""

FLUIDS = {
"CH4": {"id":"CH4","name":"Methane","Tc_K":190.564,"Pc_bar":45.992,"omega":.01142,"molar_mass_g_mol":16.0425,"triple_K":90.6941,"source":"https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=1","shomate":[(298.,1300.,(-.703029,108.4773,-42.52157,5.862788,.678565)),(1300.,6000.,(85.81217,11.26467,-2.114146,.138190,-26.42221))]},
"CO2": {"id":"CO2","name":"Carbon dioxide","Tc_K":304.1282,"Pc_bar":73.773,"omega":.22394,"molar_mass_g_mol":44.0095,"triple_K":216.592,"source":"https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=1&Units=SI","shomate":[(298.,1200.,(24.99735,55.18696,-33.69137,7.948387,-.136638)),(1200.,6000.,(58.16639,2.720074,-.492289,.038844,-6.447293))]},
"N2": {"id":"N2","name":"Nitrogen","Tc_K":126.192,"Pc_bar":33.958,"omega":.0372,"molar_mass_g_mol":28.0134,"triple_K":63.151,"source":"https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1&Units=SI","shomate":[(100.,500.,(28.98641,1.853978,-9.647459,16.63537,.000117)),(500.,2000.,(19.50583,19.88705,-8.598535,1.369784,.527601)),(2000.,6000.,(35.51872,1.128728,-.196103,.014662,-4.553760))]},
"O2": {"id":"O2","name":"Oxygen","Tc_K":154.581,"Pc_bar":50.43,"omega":.0222,"molar_mass_g_mol":31.9988,"triple_K":54.361,"source":"https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=1","shomate":[(100.,700.,(31.32234,-20.23531,57.86644,-36.50624,-.007374)),(700.,2000.,(30.03235,8.772972,-3.988133,.788313,-.741599)),(2000.,6000.,(20.91111,10.72071,-2.020498,.146449,9.245722))]}}

for _id, _name in {"CH4":"Methane","CO2":"CarbonDioxide","N2":"Nitrogen","O2":"Oxygen"}.items():
    FLUIDS[_id]["constants_source"] = "https://raw.githubusercontent.com/CoolProp/CoolProp/master/dev/fluids/"+_name+".json"

"""Pure-fluid cubic EOS calculations, using only the Python standard library.

All internal dimensions are SI: Pa, m3/mol, J/mol and J/(mol K).
Public state inputs use kelvin and bar, explicitly reflected in field names.
Reference equations: MIT 10.213 cubic-EOS handout,
https://web.mit.edu/course/10/10.213/www/handouts/cubic.pdf
and general residual-property derivations in the IDAES documentation.

P = RT/(v-b) - a(T)/[(v+delta1*b)(v+delta2*b)].
The two simple fractions make the vdW limiting case explicit. No numerical
T derivative is used for a(T); this also preserves the sign after an alpha
factor crosses zero during extreme-temperature extrapolation.
"""

import math
import sys


R = 8.31446261815324
EOS_NAMES = ("vdW", "RK", "SRK", "PR")
# delta1, delta2, Omega_a, Omega_b
_SQRT2 = math.sqrt(2.0)
MODELS = {
    "vdW": (0.0, 0.0, 27.0 / 64.0, 1.0 / 8.0),
    "RK": (1.0, 0.0, 0.42748, 0.08664),
    "SRK": (1.0, 0.0, 0.42748, 0.08664),
    "PR": (1.0 + _SQRT2, 1.0 - _SQRT2, 0.45724, 0.07780),
}


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean.")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")
    return value


def _fluid(fluid_id):
    if not isinstance(fluid_id, str) or fluid_id not in FLUIDS:
        raise ValueError("Choose CH4, CO2, N2 or O2.")
    return FLUIDS[fluid_id]


def fluids():
    """Public metadata, including the heat-capacity fit's valid intervals."""
    return [dict((k, v) for k, v in item.items() if k != "shomate") |
            {"cp_ranges_K": [[lo, hi] for lo, hi, _ in item["shomate"]]}
            for item in FLUIDS.values()]


def parameters(eos, fluid_id, T_K):
    """Return a [Pa m6/mol2], b [m3/mol] and da/dT in SI dimensions."""
    if eos not in MODELS:
        raise ValueError("Unknown EOS. Choose vdW, RK, SRK or PR.")
    f = _fluid(fluid_id)
    T = _number(T_K, "Temperature (K)", 40.0, 6000.0)
    d1, d2, oa, ob = MODELS[eos]
    tr = T / f["Tc_K"]
    ac = oa * R * R * f["Tc_K"] ** 2 / (f["Pc_bar"] * 1e5)
    b = ob * R * f["Tc_K"] / (f["Pc_bar"] * 1e5)
    if eos == "vdW":
        alpha, dalpha = 1.0, 0.0
    elif eos == "RK":
        alpha = 1.0 / math.sqrt(tr)
        dalpha = -alpha / (2.0 * T)
    else:
        w = f["omega"]
        m = (0.48 + 1.574*w - 0.176*w*w if eos == "SRK" else
             0.37464 + 1.54226*w - 0.26992*w*w)
        g = 1.0 + m * (1.0 - math.sqrt(tr))
        alpha = g * g
        dalpha = -m * g / math.sqrt(T * f["Tc_K"])
    return {"a": ac*alpha, "b": b, "da_dT": ac*dalpha,
            "alpha": alpha, "delta1": d1, "delta2": d2,
            "u": d1+d2, "w": d1*d2}


def pressure(eos, fluid_id, T_K, volume_m3_mol):
    """EOS pressure in Pa. v must exceed the model's excluded volume b."""
    p = parameters(eos, fluid_id, T_K)
    v = _number(volume_m3_mol, "Molar volume (m3/mol)", 1e-12, 1e15)
    b = p["b"]
    if v <= b:
        raise ValueError("Molar volume must exceed the excluded volume b.")
    return R * float(T_K) / (v-b) - p["a"] / ((v+p["delta1"]*b)*(v+p["delta2"]*b))


def cubic_real_roots(c2, c1, c0):
    """Distinct real roots of x^3+c2*x^2+c1*x+c0, sorted.

    Derivative stationary points partition a Cauchy root bound into monotone
    intervals. Bisection gives every sign-changing root; scale-aware residual
    tests also retain repeated roots at stationary points. The local error
    scale (not a fixed absolute tolerance) retains tiny dilute-limit roots.
    """
    coeff = (float(c2), float(c1), float(c0))
    if not all(math.isfinite(c) for c in coeff):
        raise ValueError("Cubic coefficients must be finite.")
    c2, c1, c0 = coeff
    def poly(x):
        return ((x+c2)*x+c1)*x+c0
    def zero(x):
        scale = abs(x*x*x)+abs(c2*x*x)+abs(c1*x)+abs(c0)
        return abs(poly(x)) <= 24*sys.float_info.epsilon*max(scale, 1e-300)
    bound = 1.0 + max(map(abs, coeff))
    cuts = [-bound, bound]
    discriminant = c2*c2 - 3.0*c1
    if discriminant >= 0.0:
        q = -c2 - math.copysign(math.sqrt(discriminant), c2)
        stationary = [q/3.0, c1/q] if q else [-c2/3.0]
        cuts.extend(x for x in stationary if -bound < x < bound)
    cuts = sorted(set(cuts))
    roots = [x for x in cuts[1:-1] if zero(x)]
    for lo, hi in zip(cuts, cuts[1:]):
        flo, fhi = poly(lo), poly(hi)
        # A retained stationary root belongs to both adjacent intervals.
        if zero(lo) or zero(hi) or flo*fhi >= 0.0:
            continue
        for _ in range(180):
            mid = (lo+hi)/2.0
            fm = poly(mid)
            if mid == lo or mid == hi or fm == 0.0:
                lo = hi = mid
                break
            if (flo < 0.0) == (fm < 0.0):
                lo, flo = mid, fm
            else:
                hi = mid
        roots.append((lo+hi)/2.0)
    roots.sort()
    unique = []
    for z in roots:
        if not unique or abs(z-unique[-1]) > 64*sys.float_info.epsilon*max(abs(z), abs(unique[-1]), 1e-300):
            unique.append(z)
    return unique


def _cp_segment(fluid_id, T_K):
    T = _number(T_K, "Temperature (K)", 40, 6000)
    f = _fluid(fluid_id)
    for lo, hi, coeff in f["shomate"]:
        if lo <= T <= hi:
            return coeff
    lo, hi = f["shomate"][0][0], f["shomate"][-1][1]
    raise ValueError(f"{fluid_id} heat-capacity data cover {lo:g}-{hi:g} K; total H and S are unavailable outside this range. Residual properties remain available.")


def heat_capacity(fluid_id, T_K):
    """Ideal-gas Cp in J/(mol K), with no extrapolation."""
    a, b, c, d, e = _cp_segment(fluid_id, T_K)
    t = float(T_K)/1000.0
    return a + b*t + c*t*t + d*t*t*t + e/(t*t)


def _primitive(coeff, T):
    a, b, c, d, e = coeff
    t = T/1000.0
    return (1000.0*(a*t+b*t*t/2+c*t**3/3+d*t**4/4-e/t),
            a*math.log(t)+b*t+c*t*t/2+d*t**3/3-e/(2*t*t))


def ideal_properties(fluid_id, T_K, P_bar, T_ref_K=298.15, P_ref_bar=1.0):
    """Integrate Cp dT and Cp/T dT piecewise, with -R ln(P/Pref).

    Values are zero for the ideal gas at the specified reference state. Cp
    segment boundary offsets are handled by definite integration rather than
    subtracting separately rounded tabulated integration constants.
    """
    _cp_segment(fluid_id, T_K)
    _cp_segment(fluid_id, T_ref_K)
    T, Tref = float(T_K), float(T_ref_K)
    P = _number(P_bar, "Pressure (bar)", 1e-8, 5000.0)
    Pref = _number(P_ref_bar, "Reference pressure (bar)", 1e-8, 5000.0)
    low, high = min(T, Tref), max(T, Tref)
    dh = ds = 0.0
    for lo, hi, coeff in FLUIDS[fluid_id]["shomate"]:
        left, right = max(low, lo), min(high, hi)
        if right > left:
            h0, s0 = _primitive(coeff, left)
            h1, s1 = _primitive(coeff, right)
            dh += h1-h0
            ds += s1-s0
    sign = 1.0 if T >= Tref else -1.0
    return {"H_J_mol": sign*dh, "S_J_molK": sign*ds-R*math.log(P/Pref),
            "Cp_J_molK": heat_capacity(fluid_id, T)}


def _model_state(eos, fluid_id, T, P_bar, phase):
    p = parameters(eos, fluid_id, T)
    a, b, da = p["a"], p["b"], p["da_dT"]
    P = P_bar*1e5
    A, B = a*P/(R*T)**2, b*P/(R*T)
    u, w = p["u"], p["w"]
    c2 = (u-1.0)*B-1.0
    c1 = A-u*B+(w-u)*B*B
    c0 = -A*B-w*B*B*(1.0+B)
    zs = cubic_real_roots(c2, c1, c0)
    roots = []
    for Z in zs:
        if Z <= B:
            continue
        v = Z*R*T/P
        d1, d2 = p["delta1"], p["delta2"]
        # Stable log1p avoids loss of significance as pressure tends to zero.
        J = 1.0/v if d1 == d2 else math.log1p((d1-d2)*b/(v+d2*b))/((d1-d2)*b)
        log_free = math.log(Z-B)
        hr = R*T*(Z-1.0)+(T*da-a)*J
        sr = R*log_free+da*J
        lnphi = Z-1.0-log_free-a*J/(R*T)
        phi = math.exp(lnphi) if lnphi < 700.0 else None
        den = (v+d1*b)*(v+d2*b)
        dpdv = -R*T/(v-b)**2 + a*(2*v+u*b)/den**2
        slope = dpdv*v*v/(R*T)
        mechanical = "marginal" if abs(slope) < 1e-8 else ("stable" if slope < 0 else "unstable")
        p_eos = R*T/(v-b)-a/den
        roots.append({"Z": Z, "volume_cm3_mol": v*1e6, "volume_m3_mol": v,
                      "ln_phi": lnphi, "phi": phi,
                      "fugacity_bar": None if phi is None else phi*P_bar,
                      "H_res_J_mol": hr, "S_res_J_molK": sr,
                      "H_J_mol": None, "S_J_molK": None,
                      "mechanical_stability": mechanical,
                      "dP_dv_Pa_mol_m3": dpdv,
                      "status": "unstable" if mechanical == "unstable" else "metastable",
                      "selected": False, "branch": "single",
                      "pressure_absolute_error_Pa": abs(p_eos-P),
                      "pressure_relative_error": abs(p_eos-P)/P,
                      "gibbs_identity_error_J_mol": hr-T*sr-R*T*lnphi})
    candidates = [r for r in roots if r["mechanical_stability"] != "unstable"]
    if not candidates:
        raise ValueError(f"No admissible mechanically stable {eos} root at this state.")
    if len(roots) > 1:
        roots[0]["branch"], roots[-1]["branch"] = "liquid", "vapor"
        for root in roots[1:-1]:
            root["branch"] = "intermediate"
    elif T < FLUIDS[fluid_id]["Tc_K"]:
        # Below Tc, even a sole fluid root belongs to the dense or dilute
        # branch. This lets coarse sweeps identify a transition that skips
        # the narrow three-root region altogether. These are fluid labels,
        # not a claim of stability against a solid phase.
        zc = 0.375 if eos == "vdW" else (0.3074 if eos == "PR" else 1.0/3.0)
        f = FLUIDS[fluid_id]
        vc = zc*R*f["Tc_K"]/(f["Pc_bar"]*1e5)
        roots[0]["branch"] = "liquid" if roots[0]["volume_m3_mol"] < vc else "vapor"
    equilibrium = min(candidates, key=lambda r: r["ln_phi"])
    tied = [r for r in candidates if abs(r["ln_phi"]-equilibrium["ln_phi"]) <= 1e-7]
    for root in tied:
        root["status"] = "coexisting" if len(tied) > 1 else "stable"
    selected = (candidates[-1] if phase == "vapor" else candidates[0] if phase == "liquid" else equilibrium)
    selected["selected"] = True
    return {"eos": eos, "A": A, "B": B, **p,
            "cubic_coefficients": [1.0, c2, c1, c0], "real_Z_roots": zs,
            "roots": roots, "selected": selected,
            "equilibrium_Z": equilibrium["Z"]}


def calculate(fluid_id, T_K, P_bar, phase="stable", reference="ideal", T_ref_K=298.15, P_ref_bar=1.0):
    """Compare the four EOS at one state and return JSON-ready results.

    phase='stable' minimizes molar Gibbs energy among mechanically admissible
    roots (equivalent to minimum ln(phi)); vapor/liquid select outer branches
    and explicitly preserve metastability labels. No phase fraction is inferred.
    reference='ideal': H=Hig+Hr, S=Sig+Sr, with Hig=Sig=0 at Tref,Pref.
    reference='real': subtract stable reference residuals for that EOS as well.
    """
    f = _fluid(fluid_id)
    T = _number(T_K, "Temperature (K)", 40.0, 6000.0)
    P = _number(P_bar, "Pressure (bar)", 1e-8, 5000.0)
    Tref = _number(T_ref_K, "Reference temperature (K)", 40.0, 6000.0)
    Pref = _number(P_ref_bar, "Reference pressure (bar)", 1e-8, 5000.0)
    if phase not in ("stable", "vapor", "liquid"):
        raise ValueError("Root choice must be stable, vapor or liquid.")
    if reference not in ("ideal", "real"):
        raise ValueError("Reference basis must be ideal or real.")
    warnings = []
    try:
        ideal = ideal_properties(fluid_id, T, P, Tref, Pref)
    except ValueError as exc:
        ideal = None
        warnings.append(str(exc))
    if T < f["triple_K"]:
        warnings.append("Temperature is below the fluid's triple point. Solid phases are omitted; fluid-root stability is not global phase stability.")
    if T/f["Tc_K"] > 3.0 or T/f["Tc_K"] < 0.5 or P/f["Pc_bar"] > 10.0:
        warnings.append("This state is far from the critical-property fitting region. Cubic-EOS extrapolation may be inaccurate even when the algebra is satisfied.")
    if abs(T/f["Tc_K"]-1.0) < 0.02 and abs(P/f["Pc_bar"]-1.0) < 0.1:
        warnings.append("Near-critical roots are sensitive to rounded EOS constants and input precision.")
    models = []
    for eos in EOS_NAMES:
        model = _model_state(eos, fluid_id, T, P, phase)
        offset_h = offset_s = 0.0
        if reference == "real":
            ref_model = _model_state(eos, fluid_id, Tref, Pref, "stable")
            ref_root = ref_model["selected"]
            offset_h, offset_s = ref_root["H_res_J_mol"], ref_root["S_res_J_molK"]
            model["reference_root_Z"] = ref_root["Z"]
            if ref_root["status"] == "coexisting":
                warnings.append(f"{eos}: reference has coexisting roots; the selected minimum-Gibbs reference root is reported explicitly.")
        if ideal is not None:
            for root in model["roots"]:
                root["H_J_mol"] = ideal["H_J_mol"]+root["H_res_J_mol"]-offset_h
                root["S_J_molK"] = ideal["S_J_molK"]+root["S_res_J_molK"]-offset_s
        if any(r["phi"] is None for r in model["roots"]):
            warnings.append(f"{eos}: fugacity coefficient exceeds floating-point range; ln(phi) remains available.")
        if any(r["pressure_relative_error"] > 1e-6 for r in model["roots"]):
            warnings.append(f"{eos}: pressure cancellation reduces relative numerical accuracy on a dense root at very low pressure. Per-root absolute and relative closure errors are reported.")
        if model["selected"]["status"] == "metastable":
            warnings.append(f"{eos}: requested {phase} branch is metastable; the lowest-Gibbs root is different.")
        if any(r["mechanical_stability"] == "marginal" for r in model["roots"]):
            warnings.append(f"{eos}: a root is mechanically marginal (near a spinodal or critical point).")
        if len([r for r in model["roots"] if r["status"] == "coexisting"]) > 1:
            warnings.append(f"{eos}: roots have equal fugacity within 1e-7 in ln(phi); phase fraction is undetermined.")
        models.append(model)
    return {"input": {"fluid_id": fluid_id, "T_K": T, "P_bar": P, "phase": phase,
                      "T_ref_K": Tref, "P_ref_bar": Pref},
            "fluid": next(item for item in fluids() if item["id"] == fluid_id),
            "reference": {"basis": reference, "T_K": Tref, "P_bar": Pref,
                          "description": "Ideal gas H=S=0 at reference state" if reference == "ideal" else "Stable real fluid H=S=0 at reference state, separately for each EOS"},
            "ideal": ideal, "warnings": list(dict.fromkeys(warnings)), "models": models}


def _point_count(points, low, high):
    result = _number(points, "Point count", low, high)
    if not result.is_integer():
        raise ValueError("Point count must be an integer.")
    return int(result)


def sweep(fluid_id, T_K, P_min_bar, P_max_bar, points=70, phase="stable", reference="ideal", T_ref_K=298.15, P_ref_bar=1.0):
    """Log-spaced pressure sweep; branch jumps are marked for plotting gaps."""
    count = _point_count(points, 2, 300)
    pmin = _number(P_min_bar, "Minimum pressure (bar)", 1e-8, 5000)
    pmax = _number(P_max_bar, "Maximum pressure (bar)", 1e-8, 5000)
    if pmax <= pmin:
        raise ValueError("Maximum pressure must exceed minimum pressure.")
    result = []
    previous = {}
    for i in range(count):
        P = math.exp(math.log(pmin)+(math.log(pmax)-math.log(pmin))*i/(count-1))
        P = min(pmax, max(pmin, P))
        state = calculate(fluid_id, T_K, P, phase, reference, T_ref_K, P_ref_bar)
        models = []
        for m in state["models"]:
            row = {"eos": m["eos"], **m["selected"]}
            old = previous.get(m["eos"])
            row["break_before"] = bool(old and row["branch"] != old["branch"]
                                       and row["branch"] != "single" and old["branch"] != "single")
            previous[m["eos"]] = row
            models.append(row)
        result.append({"P_bar": P, "models": models})
    return result


def isotherm(fluid_id, T_K, P_bar, points=160):
    """EOS P-v curves on a log-volume grid; these include non-equilibrium loops.

    Negative-pressure portions are preserved mathematically. They are not
    interpreted as equilibrium two-phase plateaus or experimental predictions.
    """
    count = _point_count(points, 20, 1000)
    state = calculate(fluid_id, T_K, P_bar)
    largest = max(r["volume_m3_mol"] for m in state["models"] for r in m["roots"])
    series = []
    for model in state["models"]:
        vmin = model["b"]*1.06
        vmax = max(largest*4.0, model["b"]*60)
        points_out = []
        for i in range(count):
            v = math.exp(math.log(vmin)+(math.log(vmax)-math.log(vmin))*i/(count-1))
            points_out.append({"volume_cm3_mol": v*1e6,
                               "P_bar": pressure(model["eos"], fluid_id, T_K, v)/1e5})
        series.append({"eos": model["eos"], "points": points_out})
    return {"series": series, "T_K": float(T_K), "state_P_bar": float(P_bar),
            "note": "Raw homogeneous-fluid EOS curves; loops are not equilibrium two-phase paths."}
HTML = '<!doctype html>\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <meta name="color-scheme" content="light">\n  <title>CubicLab · Real-gas thermodynamics</title>\n  <link rel="stylesheet" href="/style.css">\n  <script src="/app.js" defer></script>\n</head>\n<body>\n  <header class="topbar">\n    <a class="brand" href="/" aria-label="CubicLab home"><span class="brand-symbol" aria-hidden="true">C<span>3</span></span><span>CubicLab<small>REAL-GAS THERMODYNAMICS</small></span></a>\n    <div class="header-right"><span class="offline"><i></i> Local &amp; offline</span><a href="#methods">How it works <span aria-hidden="true">↗️</span></a></div>\n  </header>\n  <main class="workspace">\n    <section class="intro"><div><p class="eyebrow">THE CUBIC EQUATION WORKBENCH</p><h1>One state. Four perspectives.</h1><p>Explore volume, energy and fugacity with four classical equations of state.</p></div><span class="edition">vdW <span>·</span> RK <span>·</span> SRK <span>·</span> PR</span></section>\n    <div class="layout">\n      <aside class="controls panel">\n        <div class="section-heading"><span class="step">01</span><h2>Define your state</h2></div>\n        <form id="state-form">\n          <label for="fluid">Pure component</label><select id="fluid" name="fluid"><option value="CO2">Carbon dioxide · CO₂</option><option value="CH4">Methane · CH₄</option><option value="N2">Nitrogen · N₂</option><option value="O2">Oxygen · O₂</option></select>\n          <div class="fluid-meta" id="fluid-meta">Loading fluid data…</div>\n          <div class="input-pair"><div><label for="temperature">Temperature</label><div class="unit-input"><input id="temperature" name="temperature" type="number" step="any" min="0.000001" value="350" required><span>K</span></div></div><div><label for="pressure">Pressure</label><div class="unit-input pressure-input"><input id="pressure" name="pressure" type="number" step="any" min="0.000000001" value="40" required><select id="pressure-unit" aria-label="Pressure unit"><option value="bar">bar</option><option value="MPa">MPa</option></select></div></div></div>\n          <label for="phase">Root selection</label><select id="phase"><option value="stable">Stable equilibrium root</option><option value="vapor">Largest mechanically stable root</option><option value="liquid">Smallest mechanically stable root</option></select>\n          <p class="field-note">All admissible roots remain visible below.</p>\n          <details class="reference-options"><summary>Enthalpy &amp; entropy reference</summary><label for="reference">Reference convention</label><select id="reference"><option value="ideal">Ideal gas: H = S = 0 at reference</option><option value="real">Real fluid: H = S = 0 at reference</option></select><div class="input-pair"><div><label for="tref">Reference T / K</label><input id="tref" type="number" min="0.000001" step="any" value="298.15" required></div><div><label for="pref">Reference P / bar</label><input id="pref" type="number" min="0.000000001" step="any" value="1" required></div></div><p class="field-note">Real-fluid zero uses the stable root of each model. These are relative molar properties, not absolute values.</p></details>\n          <button class="primary" id="calculate" type="submit">Calculate state <span aria-hidden="true">→</span></button>\n          <p class="field-note update-note" id="update-note" aria-live="polite">Ready to explore.</p>\n        </form>\n        <div class="presets"><p class="eyebrow">TRY A STATE</p><button type="button" data-preset="co2"><span>Compressed CO₂</span><small>350 K · 40 bar</small></button><button type="button" data-preset="methane"><span>Three-root methane</span><small>150 K · 10 bar</small></button><button type="button" data-preset="nitrogen"><span>Near the ideal-gas limit</span><small>N₂ · 300 K · 0.01 bar</small></button></div>\n        <div class="sidebar-note"><span aria-hidden="true">ⓘ</span><p>Cubic EOS predictions are an educational approximation. Solid phases and phase fractions are outside this model.</p></div>\n      </aside>\n      <div class="results-column">\n        <div id="error" class="message error" role="alert" hidden></div><div id="warnings" aria-live="polite"></div>\n        <section class="state-panel panel" aria-labelledby="state-heading"><div class="result-topline"><div class="section-heading"><span class="step">02</span><h2 id="state-heading">State snapshot</h2></div><label class="focus-label" for="focus-model">Focus model <select id="focus-model"><option>PR</option><option>SRK</option><option>RK</option><option>vdW</option></select></label></div><div class="state-caption"><span id="state-label">Carbon dioxide at 350 K and 40 bar</span><span class="phase-pill" id="phase-label">Awaiting calculation</span></div><div class="metrics" id="metrics"><div class="metric"><span>Compressibility</span><strong>—</strong><small>Z · dimensionless</small></div><div class="metric"><span>Molar volume</span><strong>—</strong><small>cm³/mol</small></div><div class="metric"><span>Fugacity</span><strong>—</strong><small>bar</small></div><div class="metric"><span>Fugacity coefficient</span><strong>—</strong><small>φ · dimensionless</small></div></div><p class="reference-caption" id="reference-caption">H and S use a stated reference; residual values compare the real fluid with an ideal gas at the same T and P.</p></section>\n        <section class="panel model-panel" aria-labelledby="compare-heading"><div class="panel-heading"><div class="section-heading"><span class="step">03</span><h2 id="compare-heading">Compare the models</h2></div><button id="export-results" class="text-button" type="button" disabled>↓ Results CSV</button></div><div class="table-scroll"><table id="model-table"><thead><tr><th scope="col">Model</th><th scope="col">Z</th><th scope="col">V<small>cm³/mol</small></th><th scope="col">φ</th><th scope="col">f<small>bar</small></th><th scope="col">Hᴿ<small>J/mol</small></th><th scope="col">Sᴿ<small>J/(mol K)</small></th><th scope="col">H<small>J/mol</small></th><th scope="col">S<small>J/(mol K)</small></th></tr></thead><tbody><tr><td colspan="9" class="empty">Calculating the initial state…</td></tr></tbody></table></div><p class="table-note">Each row shows the selected root. Hᴿ and Sᴿ are residual properties; H and S include the ideal-gas contribution and chosen reference.</p></section>\n        <section class="panel plot-panel" aria-labelledby="plot-heading"><div class="panel-heading"><div class="section-heading"><span class="step">04</span><h2 id="plot-heading">See the behavior</h2></div><button id="export-plot" class="text-button" type="button" disabled>↓ Plot SVG</button></div><div class="plot-tabs" role="tablist" aria-label="Plot type"><button id="tab-pv" role="tab" aria-selected="true" aria-controls="plot-area" data-plot="pv">P–V isotherm</button><button id="tab-z" role="tab" aria-selected="false" aria-controls="plot-area" data-plot="z" tabindex="-1">Compressibility</button><button id="tab-h" role="tab" aria-selected="false" aria-controls="plot-area" data-plot="h" tabindex="-1">Residual H</button><button id="tab-s" role="tab" aria-selected="false" aria-controls="plot-area" data-plot="s" tabindex="-1">Residual S</button><button id="tab-phi" role="tab" aria-selected="false" aria-controls="plot-area" data-plot="phi" tabindex="-1">Fugacity coefficient</button></div><div class="plot-tools"><span id="plot-subtitle">Four models at constant temperature</span><div class="sweep-controls"><label for="pmin">Sweep / bar</label><input type="number" id="pmin" step="any" min="0.000000001" value="0.1" aria-label="Sweep minimum pressure in bar"><span>to</span><input type="number" id="pmax" step="any" min="0.000000001" value="100" aria-label="Sweep maximum pressure in bar"><button id="update-sweep" type="button">Update</button></div></div><div id="plot-area" role="tabpanel" aria-labelledby="tab-pv"><div class="empty plot-empty">Preparing plot…</div></div><div id="plot-legend" class="plot-legend"></div><div class="plot-bottom"><p id="plot-note">The volume axis is logarithmic. Negative pressures are mathematical extensions of the EOS.</p><button id="export-sweep" class="text-button" type="button" disabled>↓ Sweep CSV</button></div></section>\n        <section class="panel roots-panel"><div class="panel-heading"><div class="section-heading"><span class="step">05</span><h2>Inspect every root</h2></div><span class="small-tag" id="root-model-label">PR</span></div><p class="table-note" id="all-roots">Real mathematical Z roots will appear here.</p><div class="table-scroll"><table id="root-table"><thead><tr><th>Root</th><th>Z</th><th>V<small>cm³/mol</small></th><th>ln φ</th><th>Mechanical stability</th><th>Thermodynamic status</th></tr></thead><tbody><tr><td colspan="6" class="empty">No roots to display yet.</td></tr></tbody></table></div><p class="table-note">An admissible root has Z &gt; B. Stable equilibrium minimizes ln φ among mechanically stable roots. An unstable middle root is never selected.</p></section>\n        <section class="panel validation-panel"><div class="panel-heading"><div class="section-heading"><span class="step">✓</span><h2>Built-in validation</h2></div><button id="validate" class="text-button" type="button">Run checks →</button></div><div id="validation-results" class="validation-content">Verify pressure closure, ideal-gas limits, thermodynamic identities and reference consistency with the supplied test suite.</div></section>\n      </div>\n    </div>\n    <section id="methods" class="methods"><div><p class="eyebrow">A TRANSPARENT MODEL</p><h2>Understand the numbers.</h2><p>One calculation engine handles all four equations, keeping units, reference states and root selection consistent.</p></div><div class="method-grid"><article><h3>01 / The equation</h3><p class="formula">P = RT/(V − b) − a(T)/[(V + δ₁b)(V + δ₂b)]</p><p>Each model changes its attraction term, temperature dependence and volume offsets. All calculations use SI units internally; the interface converts pressure and volume for display.</p></article><article><h3>02 / Energy &amp; entropy</h3><p>For the ideal-gas reference, H = H<sup>ig</sup> + Hᴿ and S = S<sup>ig</sup> + Sᴿ. A real-fluid zero also subtracts that model’s stable reference residuals. Ideal-gas terms integrate NIST Shomate heat capacities; totals are unavailable outside supported temperature ranges.</p></article><article><h3>03 / Fugacity &amp; phase</h3><p>f = φP. At a fixed temperature and pressure, the stable root has the smallest chemical potential (equivalently ln φ). Forced small/large-volume selections can be metastable. Coexisting roots do not determine a phase fraction.</p></article><article><h3>04 / Sources &amp; scope</h3><p>Based on the classical vdW, RK, SRK and PR equations and <a href="https://pages.nist.gov/teqp-docs/en/main/models/cubics.html" target="_blank" rel="noreferrer">NIST generalized-cubic documentation</a>. Ideal-gas data: <a href="https://webbook.nist.gov/chemistry/" target="_blank" rel="noreferrer">NIST Chemistry WebBook</a>. Consult the project report for equations, tests and limitations.</p></article></div></section>\n  </main>\n  <footer><strong>CubicLab</strong><span>Pure fluids · Molar properties · Explicit references</span><span>Built for exploration, with the workings visible.</span></footer>\n</body>\n</html>'
JS = '/* CubicLab\'s presentation layer. Thermodynamics are computed only by the Python engine. */\n\'use strict\';\nconst $ = (id) => document.getElementById(id);\nconst COLORS = {\n    vdW: \'#bb8254\',\n    RK: \'#8d85a8\',\n    SRK: \'#7599ac\',\n    PR: \'#087f79\'\n};\nconst MODEL_NAMES = {\n    vdW: \'van der Waals\',\n    RK: \'Redlich–Kwong\',\n    SRK: \'Soave–Redlich–Kwong\',\n    PR: \'Peng–Robinson\'\n};\nconst state = {\n    fluids: [],\n    result: null,\n    plots: null,\n    input: null,\n    plot: \'pv\',\n    request: 0,\n    dirty: false,\n    plotRequest: 0,\n    inputRevision: 0\n};\nconst esc = (text) => String(text ?? \'\').replace(/[&<>"\']/g, c => ({\n    \'&\': \'&amp;\',\n    \'<\': \'&lt;\',\n    \'>\': \'&gt;\',\n    \'"\': \'&quot;\',\n    "\'": \'&#39;\'\n}[c]));\nconst number = (value, digits=4) => Number.isFinite(value) ? (Math.abs(value) > 0 && (Math.abs(value) < .0001 || Math.abs(value) >= 1e7) ? value.toExponential(3) : value.toLocaleString(\'en-US\', {\n    maximumFractionDigits: digits\n})) : \'—\';\nconst precise = value => Number.isFinite(value) ? value.toPrecision(12) : \'\';\nfunction titleCase(value) {\n    return String(value || \'\').replace(/[_-]/g, \' \').replace(/\\b\\w/g, c => c.toUpperCase());\n}\nasync function api(path, payload) {\n    const response = await fetch(path, payload ? {\n        method: \'POST\',\n        headers: {\n            \'Content-Type\': \'application/json\'\n        },\n        body: JSON.stringify(payload)\n    } : undefined);\n    let data;\n    try {\n        data = await response.json();\n    } catch {\n        throw new Error(\'The local server did not return a readable response. Check that app.py is running.\');\n    }\n    if (!response.ok)\n        throw new Error(typeof data.error === \'string\' ? data.error : (data.error?.message || data.message || `Request failed (${response.status}).`));\n    return data;\n}\nfunction inputValues() {\n    const values = {\n        fluid_id: $(\'fluid\').value,\n        T_K: Number($(\'temperature\').value),\n        P_bar: Number($(\'pressure\').value) * ($(\'pressure-unit\').value === \'MPa\' ? 10 : 1),\n        phase: $(\'phase\').value,\n        reference: $(\'reference\').value,\n        T_ref_K: Number($(\'tref\').value),\n        P_ref_bar: Number($(\'pref\').value)\n    };\n    for (const [name,value] of Object.entries(values)) {\n        if (typeof value === \'number\' && (!Number.isFinite(value) || value <= 0))\n            throw new Error(`${({\n                T_K: \'Temperature\',\n                P_bar: \'Pressure\',\n                T_ref_K: \'Reference temperature\',\n                P_ref_bar: \'Reference pressure\'\n            })[name]} must be a finite positive number.`);\n    }\n    return values;\n}\nfunction sweepValues() {\n    const low = Number($(\'pmin\').value)\n      , high = Number($(\'pmax\').value);\n    if (!Number.isFinite(low) || !Number.isFinite(high) || low <= 0 || high <= low)\n        throw new Error(\'Sweep pressures must be positive, with the maximum greater than the minimum.\');\n    return {\n        P_min_bar: low,\n        P_max_bar: high,\n        points: 85\n    };\n}\nfunction selectedModel() {\n    return state.result?.models.find(m => m.eos === $(\'focus-model\').value);\n}\nfunction selectedRoot(model) {\n    return model?.selected || model?.roots?.find(r => r.selected) || null;\n}\nfunction renderFluidMeta() {\n    const f = state.fluids.find(f => f.id === $(\'fluid\').value);\n    $(\'fluid-meta\').innerHTML = f ? `<span>Tc <strong>${number(f.Tc_K, 2)} K</strong></span><span>Pc <strong>${number(f.Pc_bar, 2)} bar</strong></span><span>ω <strong>${number(f.omega, 4)}</strong></span>` : \'Critical properties will load with the calculation.\';\n}\nfunction setError(message) {\n    $(\'error\').textContent = message || \'\';\n    $(\'error\').hidden = !message;\n}\nfunction markDirty() {\n    ++state.inputRevision;\n    if (!state.result)\n        return;\n    state.dirty = true;\n    document.querySelector(\'.results-column\').classList.add(\'stale\');\n    $(\'update-note\').textContent = \'Inputs changed. Calculate to update results.\';\n    $(\'export-results\').disabled = true;\n    $(\'export-sweep\').disabled = true;\n    $(\'export-plot\').disabled = true;\n}\nfunction clearResults() {\n    state.result = null;\n    state.plots = null;\n    state.input = null;\n    $(\'metrics\').querySelectorAll(\'strong\').forEach(e => e.textContent = \'—\');\n    $(\'model-table\').querySelector(\'tbody\').innerHTML = \'<tr><td colspan="9" class="empty">Correct the input and calculate again.</td></tr>\';\n    $(\'root-table\').querySelector(\'tbody\').innerHTML = \'<tr><td colspan="6" class="empty">No current results.</td></tr>\';\n    $(\'all-roots\').textContent = \'No current mathematical roots.\';\n    $(\'plot-area\').innerHTML = \'<div class="empty plot-empty">No current plot.</div>\';\n    $(\'plot-legend\').innerHTML = \'\';\n    $(\'phase-label\').textContent = \'No result\';\n    $(\'export-results\').disabled = true;\n    $(\'export-sweep\').disabled = true;\n    $(\'export-plot\').disabled = true;\n}\nasync function calculate(event) {\n    event?.preventDefault();\n    const request = ++state.request;\n    const inputRevision = state.inputRevision;\n    ++state.plotRequest;\n    setError(\'\');\n    $(\'warnings\').innerHTML = \'\';\n    document.body.classList.add(\'loading\');\n    $(\'calculate\').disabled = true;\n    $(\'calculate\').firstChild.textContent = \'Calculating… \';\n    try {\n        const input = inputValues();\n        const result = await api(\'/api/calculate\', input);\n        if (request !== state.request)\n            return;\n        state.result = result;\n        state.input = input;\n        state.plots = null;\n        state.dirty = state.inputRevision !== inputRevision;\n        document.querySelector(\'.results-column\').classList.toggle(\'stale\', state.dirty);\n        renderResults();\n        $(\'update-note\').textContent = state.dirty ? \'Inputs changed. Calculate to update results.\' : \'Calculated. All values are molar.\';\n        $(\'export-results\').disabled = state.dirty;\n        $(\'plot-area\').innerHTML = \'<div class="empty plot-empty">Preparing pressure sweep…</div>\';\n        await loadPlots();\n    } catch (error) {\n        if (request === state.request) {\n            clearResults();\n            setError(error.message);\n            $(\'update-note\').textContent = \'Calculation needs attention.\';\n        }\n    } finally {\n        if (request === state.request) {\n            document.body.classList.remove(\'loading\');\n            $(\'calculate\').disabled = false;\n            $(\'calculate\').firstChild.textContent = \'Calculate state \';\n        }\n    }\n}\nfunction renderResults() {\n    const data = state.result;\n    if (!data)\n        return;\n    const model = selectedModel()\n      , root = selectedRoot(model)\n      , input = state.input;\n    const fluid = state.fluids.find(f => f.id === input.fluid_id) || data.fluid || {};\n    $(\'state-label\').textContent = `${fluid.name || input.fluid_id} · ${number(input.T_K, 2)} K · ${number(input.P_bar, 4)} bar`;\n    $(\'phase-label\').textContent = root ? `${titleCase(root.status || root.mechanical_stability)} · ${model.eos}` : \'No selectable root\';\n    $(\'phase-label\').classList.toggle(\'warn\', !!root && root.status === \'metastable\');\n    const metrics = [[\'Compressibility\', root?.Z, 5, \'Z · dimensionless\'], [\'Molar volume\', root?.volume_cm3_mol, 2, \'cm³/mol\'], [\'Fugacity\', root?.fugacity_bar, 4, \'bar\'], [\'Fugacity coefficient\', root?.phi, 5, \'φ · dimensionless\']];\n    $(\'metrics\').innerHTML = metrics.map( ([name,value,digits,unit]) => `<div class="metric"><span>${name}</span><strong>${number(value, digits)}</strong><small>${unit}</small></div>`).join(\'\');\n    const refName = input.reference === \'real\' ? \'stable real fluid\' : \'ideal gas\';\n    $(\'reference-caption\').textContent = `Reference: H = S = 0 for the ${refName} at ${number(input.T_ref_K, 2)} K, ${number(input.P_ref_bar, 4)} bar. ${input.reference === \'real\' ? \'Each EOS has its own real-fluid zero.\' : \'Total values at the reference equal the EOS residuals.\'}`;\n    const warnings = Array.isArray(data.warnings) ? data.warnings : [];\n    $(\'warnings\').innerHTML = warnings.map(w => `<div class="message">${esc(typeof w === \'string\' ? w : w.message || JSON.stringify(w))}</div>`).join(\'\');\n    $(\'model-table\').querySelector(\'tbody\').innerHTML = data.models.map(m => {\n        const r = selectedRoot(m);\n        return `<tr class="${m.eos === $(\'focus-model\').value ? \'selected-row\' : \'\'}"><td title="${esc(MODEL_NAMES[m.eos])}"><span class="model-color" style="background:${COLORS[m.eos]}"></span>${esc(m.eos)}</td>${[[r?.Z, 5], [r?.volume_cm3_mol, 2], [r?.phi, 5], [r?.fugacity_bar, 4], [r?.H_res_J_mol, 2], [r?.S_res_J_molK, 4], [r?.H_J_mol, 2], [r?.S_J_molK, 4]].map( ([v,d]) => `<td title="${Number.isFinite(v) ? esc(v) : \'Unavailable: see the temperature-range warning\'}">${number(v, d)}</td>`).join(\'\')}</tr>`;\n    }\n    ).join(\'\');\n    $(\'root-model-label\').textContent = model?.eos || \'\';\n    $(\'all-roots\').textContent = `All real mathematical Z roots: ${(model?.real_Z_roots || []).map(z => number(z, 8)).join(\', \')}. B = ${number(model?.B, 8)}. The table below includes only admissible roots (Z > B).`;\n    $(\'root-table\').querySelector(\'tbody\').innerHTML = (model?.roots || []).map( (r, i) => `<tr class="${r.selected ? \'root-selected\' : \'\'}"><td>${i + 1}${r.selected ? \'<span class="selected-dot">selected</span>\' : \'\'}</td><td>${number(r.Z, 7)}</td><td>${number(r.volume_cm3_mol, 3)}</td><td>${number(r.ln_phi, 7)}</td><td class="${r.mechanical_stability === \'unstable\' ? \'status-bad\' : \'\'}">${esc(titleCase(r.mechanical_stability))}</td><td class="${r.status === \'unstable\' ? \'status-bad\' : \'\'}">${esc(titleCase(r.status))}</td></tr>`).join(\'\');\n}\nasync function loadPlots() {\n    if (!state.input)\n        return;\n    const plotRequest = ++state.plotRequest;\n    $(\'update-sweep\').disabled = true;\n    try {\n        const plots = await api(\'/api/plots\', {\n            ...state.input,\n            ...sweepValues()\n        });\n        if (plotRequest !== state.plotRequest)\n            return;\n        state.plots = plots;\n        renderPlot();\n        $(\'export-sweep\').disabled = state.dirty;\n        $(\'export-plot\').disabled = state.dirty;\n    } catch (error) {\n        if (plotRequest === state.plotRequest) {\n            state.plots = null;\n            setError(error.message);\n            $(\'plot-area\').innerHTML = \'<div class="empty plot-empty">Plot unavailable. Adjust the sweep range and update.</div>\';\n            $(\'export-sweep\').disabled = true;\n            $(\'export-plot\').disabled = true;\n        }\n    } finally {\n        if (plotRequest === state.plotRequest)\n            $(\'update-sweep\').disabled = false;\n    }\n}\nfunction linearTicks(min, max, count=5) {\n    const raw = (max - min) / count\n      , power = 10 ** Math.floor(Math.log10(raw || 1))\n      , scaled = raw / power\n      , step = (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;\n    const ticks = [];\n    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-8 && ticks.length < 20; v += step)\n        ticks.push(Math.abs(v) < step * 1e-8 ? 0 : v);\n    return ticks;\n}\nfunction tickText(v) {\n    if (v === 0)\n        return \'0\';\n    if (Math.abs(v) >= 1e5 || Math.abs(v) < .001)\n        return v.toExponential(1);\n    return number(v, Math.abs(v) < 1 ? 3 : 1);\n}\nfunction renderPlot() {\n    if (!state.plots || !state.input)\n        return;\n    const W = 870\n      , H = 345\n      , left = 78\n      , right = 29\n      , top = 23\n      , bottom = 57\n      , pw = W - left - right\n      , ph = H - top - bottom;\n    const plot = state.plot\n      , isPV = plot === \'pv\';\n    const meta = {\n        pv: {\n            y: \'Pressure / bar\',\n            x: \'Molar volume / cm³ mol⁻¹ (log scale)\'\n        },\n        z: {\n            y: \'Compressibility factor, Z\',\n            key: \'Z\'\n        },\n        h: {\n            y: \'Residual enthalpy / J mol⁻¹\',\n            key: \'H_res_J_mol\'\n        },\n        s: {\n            y: \'Residual entropy / J mol⁻¹ K⁻¹\',\n            key: \'S_res_J_molK\'\n        },\n        phi: {\n            y: \'Fugacity coefficient, φ\',\n            key: \'phi\'\n        }\n    }[plot];\n    let series;\n    if (isPV)\n        series = (state.plots.isotherm?.series || []).map(s => ({\n            eos: s.eos,\n            points: s.points.map(p => [p.volume_cm3_mol, p.P_bar])\n        }));\n    else\n        series = [\'vdW\', \'RK\', \'SRK\', \'PR\'].map(eos => ({\n            eos,\n            points: (state.plots.sweep || []).map(row => {\n                const m = row.models?.find(m => m.eos === eos);\n                const point = [row.P_bar, m?.[meta.key]];\n                point.breakBefore = !!m?.break_before;\n                return point;\n            }\n            )\n        }));\n    const finite = series.flatMap(s => s.points).filter(p => p.every(Number.isFinite) && (!isPV || p[0] > 0));\n    if (!finite.length) {\n        $(\'plot-area\').innerHTML = \'<div class="empty plot-empty">No finite data to plot.</div>\';\n        return;\n    }\n    let xmin = Math.min(...finite.map(p => p[0]))\n      , xmax = Math.max(...finite.map(p => p[0]));\n    let ymin = Math.min(...finite.map(p => p[1]))\n      , ymax = Math.max(...finite.map(p => p[1]));\n    if (isPV) {\n        const fluid = state.fluids.find(f => f.id === state.input.fluid_id);\n        ymax = Math.max(2 * state.input.P_bar, 1.4 * (fluid?.Pc_bar || 50), 10);\n        ymin = Math.max(Math.min(ymin, 0), -.3 * ymax);\n    } else {\n        const margin = (ymax - ymin) * .12 || Math.max(Math.abs(ymax) * .05, .02);\n        ymin -= margin;\n        ymax += margin;\n        if (plot === \'z\' || plot === \'phi\') {\n            ymin = Math.min(ymin, 1);\n            ymax = Math.max(ymax, 1);\n        }\n    }\n    if (xmax === xmin)\n        xmax = xmin + 1;\n    if (ymax === ymin)\n        ymax = ymin + 1;\n    const sx = x => left + ((isPV ? Math.log(x / xmin) : x - xmin) / (isPV ? Math.log(xmax / xmin) : xmax - xmin)) * pw;\n    const sy = y => top + (ymax - y) / (ymax - ymin) * ph;\n    const xTicks = isPV ? ( () => {\n        const t = [];\n        for (let p = Math.floor(Math.log10(xmin)); p <= Math.ceil(Math.log10(xmax)); p++) {\n            for (const m of [1, 2, 5]) {\n                const v = m * 10 ** p;\n                if (v >= xmin && v <= xmax)\n                    t.push(v);\n            }\n        }\n        return t.length > 9 ? t.filter( (_, i) => i % 2 === 0) : t;\n    }\n    )() : linearTicks(xmin, xmax, 6);\n    const yTicks = linearTicks(ymin, ymax, 5);\n    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">${esc(meta.y)} for ${esc(state.input.fluid_id)} at ${number(state.input.T_K)} K</title><desc id="chart-desc">Comparison of van der Waals, Redlich-Kwong, Soave-Redlich-Kwong and Peng-Robinson equations. ${isPV ? \'Logarithmic molar-volume axis.\' : \'Pressure sweep in bar.\'}</desc><defs><clipPath id="plot-clip"><rect x="${left}" y="${top}" width="${pw}" height="${ph}"/></clipPath></defs><rect width="${W}" height="${H}" fill="#ffffff"/>`;\n    for (const y of yTicks)\n        svg += `<line x1="${left}" x2="${W - right}" y1="${sy(y)}" y2="${sy(y)}" stroke="#e9eee6" stroke-width="1"/><text x="${left - 12}" y="${sy(y) + 3}" text-anchor="end" fill="#91a08e" font-size="10">${esc(tickText(y))}</text>`;\n    for (const x of xTicks)\n        svg += `<line x1="${sx(x)}" x2="${sx(x)}" y1="${top}" y2="${H - bottom}" stroke="#f1f4ee" stroke-width="1"/><text x="${sx(x)}" y="${H - bottom + 21}" text-anchor="middle" fill="#91a08e" font-size="10">${esc(tickText(x))}</text>`;\n    svg += `<line x1="${left}" x2="${W - right}" y1="${H - bottom}" y2="${H - bottom}" stroke="#cdd9cd"/><text x="${left + pw / 2}" y="${H - 10}" text-anchor="middle" fill="#6d856d" font-size="10">${esc(meta.x || \'Pressure / bar\')}</text><text transform="translate(18 ${top + ph / 2}) rotate(-90)" text-anchor="middle" fill="#6d856d" font-size="10">${esc(meta.y)}</text><g clip-path="url(#plot-clip)">`;\n    if ((plot === \'z\' || plot === \'phi\') && ymin < 1 && ymax > 1)\n        svg += `<line x1="${left}" x2="${W - right}" y1="${sy(1)}" y2="${sy(1)}" stroke="#b7c8af" stroke-dasharray="5 4"/>`;\n    if (isPV)\n        svg += `<line x1="${left}" x2="${W - right}" y1="${sy(state.input.P_bar)}" y2="${sy(state.input.P_bar)}" stroke="#c4a986" stroke-dasharray="5 4"/><text x="${W - right - 5}" y="${sy(state.input.P_bar) - 7}" text-anchor="end" font-size="9" fill="#aa946f">input pressure</text>`;\n    for (const s of series) {\n        let path = \'\'\n          , prev = null;\n        for (const p of s.points) {\n            if (!p.every(Number.isFinite) || (isPV && p[0] <= 0)) {\n                prev = null;\n                continue;\n            }\n            const x = sx(p[0])\n              , y = sy(p[1]);\n            path += `${!prev || p.breakBefore ? \'M\' : \'L\'}${x.toFixed(2)},${y.toFixed(2)} `;\n            prev = p;\n        }\n        svg += `<path d="${path}" fill="none" stroke="${COLORS[s.eos] || \'#555\'}" stroke-width="${s.eos === $(\'focus-model\').value ? 2.7 : 1.8}" stroke-linejoin="round" stroke-linecap="round"><title>${esc(MODEL_NAMES[s.eos] || s.eos)}</title></path>`;\n    }\n    if (isPV && state.result) {\n        for (const model of state.result.models) {\n            const r = selectedRoot(model);\n            if (r && r.volume_cm3_mol >= xmin && r.volume_cm3_mol <= xmax)\n                svg += `<circle cx="${sx(r.volume_cm3_mol)}" cy="${sy(state.input.P_bar)}" r="3.4" fill="${COLORS[model.eos]}" stroke="white" stroke-width="1.3"><title>${esc(model.eos)} selected root: ${number(r.volume_cm3_mol, 3)} cm³/mol</title></circle>`;\n        }\n    }\n    svg += \'</g><g id="chart-cursor" visibility="hidden"><line id="cursor-line" y1="\' + top + \'" y2="\' + (H - bottom) + \'" stroke="#a3b6a3" stroke-dasharray="3 3"/><rect id="cursor-box" x="0" y="0" width="180" height="94" rx="6" fill="#173f3c" opacity=".95"/><text id="cursor-text" x="0" y="0" fill="white" font-size="10"></text></g></svg>\';\n    $(\'plot-area\').innerHTML = svg;\n    $(\'plot-subtitle\').textContent = `${state.input.fluid_id} at ${number(state.input.T_K, 2)} K · ${isPV ? \'selected roots marked\' : \'selected branch; phase jumps shown as gaps\'}`;\n    $(\'plot-note\').textContent = isPV ? \'Logarithmic volume axis. The pressure window clips the high-pressure pole. Negative pressures are mathematical extensions.\' : \'Each point uses the chosen root policy. Phase changes may cause discontinuities; no two-phase mixture is implied.\';\n    $(\'plot-legend\').innerHTML = series.map(s => `<span><i class="legend-line" style="background:${COLORS[s.eos]}"></i>${esc(s.eos)}</span>`).join(\'\');\n    const svgNode = $(\'plot-area\').querySelector(\'svg\');\n    svgNode.addEventListener(\'pointermove\', event => {\n        const box = svgNode.getBoundingClientRect()\n          , px = (event.clientX - box.left) * W / box.width;\n        if (px < left || px > W - right)\n            return $(\'chart-cursor\').setAttribute(\'visibility\', \'hidden\');\n        const xval = isPV ? xmin * Math.exp((px - left) / pw * Math.log(xmax / xmin)) : xmin + (px - left) / pw * (xmax - xmin);\n        const lines = [`${number(xval, 2)} ${isPV ? \'cm³/mol\' : \'bar\'}`];\n        for (const s of series) {\n            let nearest = null\n              , dist = Infinity;\n            for (const p of s.points) {\n                if (p.every(Number.isFinite)) {\n                    const d = Math.abs(sx(p[0]) - px);\n                    if (d < dist) {\n                        dist = d;\n                        nearest = p;\n                    }\n                }\n            }\n            if (nearest)\n                lines.push(`${s.eos}: ${number(nearest[1], 4)}`);\n        }\n        $(\'chart-cursor\').setAttribute(\'visibility\', \'visible\');\n        $(\'cursor-line\').setAttribute(\'x1\', px);\n        $(\'cursor-line\').setAttribute(\'x2\', px);\n        const tx = Math.min(Math.max(px + 12, left + 2), W - right - 182)\n          , ty = top + 7;\n        $(\'cursor-box\').setAttribute(\'x\', tx);\n        $(\'cursor-box\').setAttribute(\'y\', ty);\n        $(\'cursor-text\').innerHTML = lines.map( (line, i) => `<tspan x="${tx + 11}" y="${ty + 17 + i * 16}">${esc(line)}</tspan>`).join(\'\');\n    }\n    );\n    svgNode.addEventListener(\'pointerleave\', () => $(\'chart-cursor\')?.setAttribute(\'visibility\', \'hidden\'));\n}\nfunction download(filename, content, type) {\n    const blob = new Blob([content],{\n        type\n    })\n      , url = URL.createObjectURL(blob)\n      , a = document.createElement(\'a\');\n    a.href = url;\n    a.download = filename;\n    document.body.appendChild(a);\n    a.click();\n    a.remove();\n    setTimeout( () => URL.revokeObjectURL(url), 1000);\n}\nfunction csv(rows) {\n    return rows.map(row => row.map(v => {\n        const s = String(v ?? \'\');\n        return /[",\\r\\n]/.test(s) ? `"${s.replace(/"/g, \'""\')}"` : s;\n    }\n    ).join(\',\')).join(\'\\r\\n\');\n}\nfunction exportResults() {\n    if (!state.result || state.dirty)\n        return;\n    const i = state.input\n      , rows = [[\'fluid\', \'T_K\', \'P_bar\', \'reference\', \'T_ref_K\', \'P_ref_bar\', \'root_policy\', \'EOS\', \'root\', \'selected\', \'status\', \'mechanical_stability\', \'Z\', \'V_cm3_mol\', \'ln_phi\', \'phi\', \'fugacity_bar\', \'H_res_J_mol\', \'S_res_J_molK\', \'H_J_mol\', \'S_J_molK\']];\n    for (const m of state.result.models)\n        for (const [k,r] of m.roots.entries())\n            rows.push([i.fluid_id, i.T_K, i.P_bar, i.reference, i.T_ref_K, i.P_ref_bar, i.phase, m.eos, k + 1, r.selected, r.status, r.mechanical_stability, ...[\'Z\', \'volume_cm3_mol\', \'ln_phi\', \'phi\', \'fugacity_bar\', \'H_res_J_mol\', \'S_res_J_molK\', \'H_J_mol\', \'S_J_molK\'].map(key => precise(r[key]))]);\n    download(`cubiclab-${i.fluid_id}-${i.T_K}K-results.csv`, csv(rows), \'text/csv;charset=utf-8\');\n}\nfunction exportSweep() {\n    if (!state.plots || state.dirty)\n        return;\n    const i = state.input\n      , rows = [[\'fluid\', \'T_K\', \'P_bar\', \'reference\', \'T_ref_K\', \'P_ref_bar\', \'root_policy\', \'EOS\', \'Z\', \'V_cm3_mol\', \'phi\', \'fugacity_bar\', \'H_res_J_mol\', \'S_res_J_molK\', \'H_J_mol\', \'S_J_molK\', \'status\']];\n    for (const row of state.plots.sweep)\n        for (const m of row.models)\n            rows.push([i.fluid_id, i.T_K, row.P_bar, i.reference, i.T_ref_K, i.P_ref_bar, i.phase, m.eos, ...[\'Z\', \'volume_cm3_mol\', \'phi\', \'fugacity_bar\', \'H_res_J_mol\', \'S_res_J_molK\', \'H_J_mol\', \'S_J_molK\'].map(key => precise(m[key])), m.status]);\n    download(`cubiclab-${i.fluid_id}-${i.T_K}K-sweep.csv`, csv(rows), \'text/csv;charset=utf-8\');\n}\nfunction exportPlot() {\n    const svg = $(\'plot-area\').querySelector(\'svg\');\n    if (!svg || state.dirty)\n        return;\n    const clone = svg.cloneNode(true);\n    clone.querySelector(\'#chart-cursor\')?.remove();\n    const title = `CubicLab · ${state.input.fluid_id} · ${number(state.input.T_K, 2)} K · ${$(\'tab-\' + state.plot).textContent}`;\n    const legend = [\'vdW\', \'RK\', \'SRK\', \'PR\'].map( (eos, i) => `<line x1="${265 + i * 90}" x2="${283 + i * 90}" y1="393" y2="393" stroke="${COLORS[eos]}" stroke-width="2.5"/><text x="${289 + i * 90}" y="397" fill="#506963" font-size="11">${eos}</text>`).join(\'\');\n    clone.setAttribute(\'viewBox\', \'0 0 870 415\');\n    clone.setAttribute(\'width\', \'870\');\n    clone.setAttribute(\'height\', \'415\');\n    clone.setAttribute(\'font-family\', \'Arial,sans-serif\');\n    clone.innerHTML = `<rect width="870" height="415" fill="white"/><text x="435" y="22" text-anchor="middle" fill="#173f3c" font-family="Arial,sans-serif" font-size="14">${esc(title)}</text><g transform="translate(0 32)">${clone.innerHTML}</g><g font-family="Arial,sans-serif">${legend}</g>`;\n    download(`cubiclab-${state.input.fluid_id}-${state.input.T_K}K-${state.plot}.svg`, new XMLSerializer().serializeToString(clone), \'image/svg+xml;charset=utf-8\');\n}\nasync function validate() {\n    $(\'validate\').disabled = true;\n    $(\'validation-results\').textContent = \'Running the reproducible validation suite…\';\n    try {\n        const result = await api(\'/api/validation\');\n        const checks = result.checks || result.tests || result.results || [];\n        if (Array.isArray(checks) && checks.length) {\n            const pass = c => c.passed === true || c.pass === true || c.status === \'passed\' || c.status === \'PASS\';\n            const passed = checks.filter(pass).length\n              , failed = checks.length - passed;\n            $(\'validation-results\').innerHTML = `<div class="check-summary ${failed ? \'fail\' : \'\'}">${failed ? \'Attention required\' : \'✓ All checks passed\'} · ${passed}/${checks.length}</div><ul class="checks-list">${checks.map(c => `<li class="${pass(c) ? \'\' : \'failed\'}"><span class="check-icon">${pass(c) ? \'✓\' : \'!\'}</span><span>${esc(c.name || c.test || c.description)}${c.detail ? `<small> · ${esc(typeof c.detail === \'string\' ? c.detail : JSON.stringify(c.detail))}</small>` : \'\'}</span></li>`).join(\'\')}</ul>`;\n        } else {\n            $(\'validation-results\').innerHTML = `<div class="check-summary">${esc(result.summary || result.message || \'Validation response received\')}</div><details><summary>View result details</summary><pre>${esc(JSON.stringify(result, null, 2))}</pre></details>`;\n        }\n    } catch (error) {\n        $(\'validation-results\').textContent = `Validation unavailable: ${error.message} Run the supplied Python test suite from the project folder.`;\n    } finally {\n        $(\'validate\').disabled = false;\n    }\n}\nfunction setPreset(name) {\n    const presets = {\n        co2: [\'CO2\', 350, 40],\n        methane: [\'CH4\', 150, 10],\n        nitrogen: [\'N2\', 300, .01]\n    };\n    const [fluid,T,P] = presets[name];\n    $(\'fluid\').value = fluid;\n    $(\'temperature\').value = T;\n    $(\'pressure\').value = P;\n    $(\'pressure-unit\').value = \'bar\';\n    $(\'phase\').value = \'stable\';\n    renderFluidMeta();\n    calculate();\n}\nfunction selectPlot(button) {\n    state.plot = button.dataset.plot;\n    document.querySelectorAll(\'[data-plot]\').forEach(b => {\n        b.setAttribute(\'aria-selected\', b === button ? \'true\' : \'false\');\n        b.tabIndex = b === button ? 0 : -1;\n    }\n    );\n    $(\'plot-area\').setAttribute(\'aria-labelledby\', button.id);\n    renderPlot();\n}\n$(\'state-form\').addEventListener(\'submit\', calculate);\n$(\'state-form\').querySelectorAll(\'input,select\').forEach(input => input.addEventListener(\'input\', markDirty));\n$(\'fluid\').addEventListener(\'change\', renderFluidMeta);\nlet previousPressureUnit = \'bar\';\n$(\'pressure-unit\').addEventListener(\'change\', () => {\n    const next = $(\'pressure-unit\').value;\n    const p = Number($(\'pressure\').value);\n    if (Number.isFinite(p))\n        $(\'pressure\').value = Number((p * (previousPressureUnit === \'bar\' && next === \'MPa\' ? .1 : previousPressureUnit === \'MPa\' && next === \'bar\' ? 10 : 1)).toPrecision(12));\n    previousPressureUnit = next;\n    markDirty();\n}\n);\n$(\'focus-model\').addEventListener(\'change\', () => {\n    renderResults();\n    renderPlot();\n}\n);\ndocument.querySelectorAll(\'[data-preset]\').forEach(b => b.addEventListener(\'click\', () => {\n    previousPressureUnit = \'bar\';\n    setPreset(b.dataset.preset);\n}\n));\ndocument.querySelectorAll(\'[data-plot]\').forEach(b => {\n    b.addEventListener(\'click\', () => selectPlot(b));\n    b.addEventListener(\'keydown\', event => {\n        if (![\'ArrowLeft\', \'ArrowRight\', \'Home\', \'End\'].includes(event.key))\n            return;\n        event.preventDefault();\n        const tabs = [...document.querySelectorAll(\'[data-plot]\')];\n        let i = tabs.indexOf(b);\n        i = event.key === \'Home\' ? 0 : event.key === \'End\' ? tabs.length - 1 : (i + (event.key === \'ArrowRight\' ? 1 : -1) + tabs.length) % tabs.length;\n        tabs[i].focus();\n        selectPlot(tabs[i]);\n    }\n    );\n}\n);\n$(\'update-sweep\').addEventListener(\'click\', () => {\n    setError(\'\');\n    loadPlots();\n}\n);\n$(\'export-results\').addEventListener(\'click\', exportResults);\n$(\'export-sweep\').addEventListener(\'click\', exportSweep);\n$(\'export-plot\').addEventListener(\'click\', exportPlot);\n$(\'validate\').addEventListener(\'click\', validate);\n(async () => {\n    try {\n        const data = await api(\'/api/fluids\');\n        state.fluids = data.fluids || data;\n        const current = $(\'fluid\').value;\n        $(\'fluid\').innerHTML = state.fluids.map(f => `<option value="${esc(f.id)}">${esc(f.name)} · ${esc(f.id)}</option>`).join(\'\');\n        $(\'fluid\').value = current;\n        renderFluidMeta();\n        await calculate();\n    } catch (error) {\n        clearResults();\n        setError(error.message);\n    }\n}\n)();'
CSS = ':root {\n    --ink: #172c3a;\n    --muted: #66767b;\n    --teal: #087f79;\n    --teal-pale: #e6f3ef;\n    --paper: #f6f5f0;\n    --line: #e1e5e0;\n    --white: #fff;\n    --orange: #bc7446;\n    --radius: 14px;\n    --shadow: 0 3px 18px #203c4210;\n    font-family: Inter,"Segoe UI",Arial,sans-serif;\n    color: var(--ink);\n    background: var(--paper);\n    font-synthesis: none\n}\n\n* {\n    box-sizing: border-box\n}\n\nbody {\n    margin: 0;\n    font-size: 14px;\n    line-height: 1.5\n}\n\nbutton,input,select {\n    font: inherit\n}\n\nbutton,a,input,select,summary {\n    -webkit-tap-highlight-color: transparent\n}\n\nbutton {\n    cursor: pointer\n}\n\nbutton:disabled {\n    cursor: default;\n    opacity: .45\n}\n\na {\n    color: var(--teal)\n}\n\nbutton:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible {\n    outline: 3px solid #e2a872;\n    outline-offset: 3px\n}\n\nbutton {\n    transition: background .15s,border-color .15s\n}\n\nbutton:hover:not(:disabled) {\n    filter: brightness(.96)\n}\n\n.topbar {\n    height: 87px;\n    background: #fff;\n    border-bottom: 1px solid var(--line);\n    padding: 0 max(32px,calc((100vw - 1536px)/2));\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    gap: 20px\n}\n\n.brand {\n    text-decoration: none;\n    color: var(--ink);\n    display: flex;\n    align-items: center;\n    gap: 12px;\n    font-size: 24px;\n    font-weight: 750;\n    letter-spacing: -.8px;\n    line-height: 1.1\n}\n\n.brand small {\n    display: block;\n    font-size: 9px;\n    letter-spacing: 1.6px;\n    font-weight: 600;\n    color: var(--muted);\n    margin-top: 7px\n}\n\n.brand-symbol {\n    display: block;\n    color: var(--teal);\n    font-size: 31px;\n    border: 1px solid #c2d9d4;\n    border-radius: 11px;\n    padding: 8px 10px;\n    font-weight: 500;\n    background: #f1f8f5;\n    letter-spacing: -3px\n}\n\n.brand-symbol span {\n    vertical-align: super;\n    font-size: 14px;\n    margin-left: 2px\n}\n\n.header-right {\n    display: flex;\n    align-items: center;\n    gap: 28px;\n    font-size: 12px\n}\n\n.header-right>a {\n    text-decoration: none;\n    color: var(--ink)\n}\n\n.offline {\n    color: var(--muted);\n    display: flex;\n    gap: 7px;\n    align-items: center\n}\n\n.offline i {\n    width: 6px;\n    height: 6px;\n    background: var(--teal);\n    border-radius: 50%;\n    display: inline-block\n}\n\n.workspace {\n    max-width: 1536px;\n    margin: auto;\n    padding: 32px\n}\n\n.intro {\n    display: flex;\n    justify-content: space-between;\n    align-items: end;\n    padding: 2px 0 29px\n}\n\n.eyebrow {\n    font-size: 10px;\n    letter-spacing: 1.6px;\n    font-weight: 750;\n    margin: 0 0 9px;\n    color: var(--teal)\n}\n\nh1 {\n    font-size: 31px;\n    letter-spacing: -1.1px;\n    line-height: 1.2;\n    margin: 0 0 9px;\n    font-weight: 650\n}\n\n.intro>div>p:last-child {\n    margin: 0;\n    color: var(--muted);\n    font-size: 13px\n}\n\n.edition {\n    font-size: 12px;\n    color: #6c7e81;\n    border: 1px solid #d8e0d9;\n    border-radius: 30px;\n    padding: 8px 16px;\n    white-space: nowrap;\n    margin-bottom: 3px\n}\n\n.edition span {\n    margin: 0 7px;\n    color: #9caeaa\n}\n\n.layout {\n    display: grid;\n    grid-template-columns: 300px minmax(0,1fr);\n    gap: 23px;\n    align-items: start\n}\n\n.panel {\n    background: var(--white);\n    border: 1px solid var(--line);\n    border-radius: var(--radius);\n    box-shadow: var(--shadow)\n}\n\n.controls {\n    padding: 23px;\n    position: sticky;\n    top: 18px\n}\n\n.section-heading {\n    display: flex;\n    align-items: center;\n    gap: 10px\n}\n\n.step {\n    font-size: 10px;\n    font-weight: 700;\n    letter-spacing: .6px;\n    line-height: 24px;\n    text-align: center;\n    width: 26px;\n    height: 24px;\n    border-radius: 6px;\n    background: var(--teal-pale);\n    color: var(--teal)\n}\n\nh2 {\n    font-size: 15px;\n    letter-spacing: -.2px;\n    line-height: 1.35;\n    margin: 0;\n    font-weight: 650\n}\n\n.controls .section-heading {\n    margin-bottom: 24px\n}\n\nlabel {\n    display: block;\n    font-size: 11px;\n    font-weight: 650;\n    margin: 17px 0 7px;\n    color: #40575c\n}\n\nselect,input {\n    border: 1px solid #d9e1dc;\n    border-radius: 7px;\n    background: #fbfcfa;\n    padding: 10px 11px;\n    color: var(--ink);\n    width: 100%;\n    min-width: 0;\n    outline: none;\n    height: 42px\n}\n\nselect:hover,input:hover {\n    border-color: #a9c3bb\n}\n\ninput:focus,select:focus {\n    border-color: var(--teal);\n    background: #fff\n}\n\n.fluid-meta {\n    font-size: 10px;\n    display: flex;\n    justify-content: space-between;\n    margin-top: 10px;\n    color: #7a8989;\n    gap: 5px\n}\n\n.fluid-meta strong {\n    font-weight: 600;\n    color: #415c61\n}\n\n.input-pair {\n    display: grid;\n    grid-template-columns: 1fr 1fr;\n    gap: 12px\n}\n\n.unit-input {\n    position: relative\n}\n\n.unit-input>input {\n    padding-right: 32px\n}\n\n.unit-input>span {\n    position: absolute;\n    right: 12px;\n    top: 11px;\n    color: #8a9693;\n    font-size: 12px\n}\n\n.pressure-input>input {\n    padding-right: 59px\n}\n\n.pressure-input>select {\n    width: 60px;\n    position: absolute;\n    right: 0;\n    top: 0;\n    border: 0;\n    border-left: 1px solid #d9e1dc;\n    background: transparent;\n    border-radius: 0 7px 7px 0;\n    padding: 8px 4px;\n    color: #667e7d;\n    font-size: 11px\n}\n\n.field-note {\n    font-size: 10px;\n    color: #87928e;\n    line-height: 1.55;\n    margin: 8px 0 0\n}\n\n.reference-options {\n    margin: 20px 0;\n    border-top: 1px solid var(--line);\n    padding-top: 16px\n}\n\n.reference-options summary {\n    font-size: 11px;\n    font-weight: 600;\n    cursor: pointer;\n    color: #486360;\n    list-style: none\n}\n\n.reference-options summary:after {\n    content: \'+\';\n    float: right;\n    color: var(--teal);\n    font-size: 15px;\n    line-height: 1\n}\n\n.reference-options[open] summary:after {\n    content: \'−\'\n}\n\n.reference-options select {\n    font-size: 10px;\n    padding: 7px\n}\n\n.primary {\n    width: 100%;\n    border: 0;\n    border-radius: 8px;\n    color: #fff;\n    background: var(--teal);\n    height: 43px;\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    padding: 0 15px;\n    font-size: 12px;\n    font-weight: 650;\n    box-shadow: 0 3px 8px #087f7920\n}\n\n.primary span {\n    font-size: 19px;\n    font-weight: 400\n}\n\n.update-note {\n    min-height: 16px;\n    text-align: center\n}\n\n.presets {\n    margin: 24px -23px 0;\n    padding: 21px 23px 0;\n    border-top: 1px solid var(--line)\n}\n\n.presets .eyebrow {\n    font-size: 9px;\n    color: #87948e;\n    letter-spacing: 1.1px;\n    margin-bottom: 10px\n}\n\n.presets button {\n    display: flex;\n    flex-direction: column;\n    align-items: flex-start;\n    width: 100%;\n    border: 0;\n    border-bottom: 1px solid #edf0e9;\n    background: transparent;\n    padding: 10px 0;\n    color: #435f61;\n    text-align: left;\n    font-size: 11px\n}\n\n.presets button:last-child {\n    border-bottom: 0\n}\n\n.presets button small {\n    font-size: 10px;\n    color: #98a19b;\n    margin-top: 3px\n}\n\n.sidebar-note {\n    display: flex;\n    align-items: flex-start;\n    gap: 8px;\n    margin-top: 22px;\n    color: #88948d;\n    font-size: 10px\n}\n\n.sidebar-note>span {\n    font-size: 14px;\n    color: #83a299\n}\n\n.sidebar-note p {\n    margin: 0\n}\n\n.results-column {\n    display: flex;\n    flex-direction: column;\n    gap: 20px;\n    min-width: 0\n}\n\n.state-panel {\n    padding: 21px 23px 17px\n}\n\n.result-topline,.panel-heading {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    gap: 12px\n}\n\n.focus-label {\n    display: flex;\n    align-items: center;\n    gap: 10px;\n    margin: 0;\n    font-size: 10px;\n    font-weight: 500;\n    color: #88948e\n}\n\n.focus-label select {\n    width: 72px;\n    height: 31px;\n    padding: 3px 9px;\n    font-size: 11px;\n    font-weight: 650;\n    background: #f5f8f3\n}\n\n.state-caption {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    margin-top: 19px;\n    font-size: 11px;\n    color: #758681;\n    gap: 10px\n}\n\n.phase-pill {\n    background: #edf5ef;\n    color: #5e8672;\n    font-size: 9px;\n    border-radius: 20px;\n    padding: 4px 9px;\n    white-space: nowrap\n}\n\n.phase-pill.warn {\n    background: #fff1de;\n    color: #976523\n}\n\n.metrics {\n    display: grid;\n    grid-template-columns: repeat(4,1fr);\n    margin: 21px 0 19px\n}\n\n.metric {\n    border-right: 1px solid var(--line);\n    padding-left: 20px\n}\n\n.metric:first-child {\n    padding-left: 0\n}\n\n.metric:last-child {\n    border-right: 0\n}\n\n.metric>span {\n    display: block;\n    font-size: 10px;\n    color: #7f8e87;\n    margin-bottom: 5px\n}\n\n.metric strong {\n    font-size: 29px;\n    letter-spacing: -1px;\n    font-weight: 550;\n    color: #173b3e;\n    font-variant-numeric: tabular-nums;\n    line-height: 1.3\n}\n\n.metric:first-child strong {\n    color: var(--teal)\n}\n\n.metric small {\n    display: block;\n    font-size: 9px;\n    color: #87988e;\n    margin-top: 6px;\n    letter-spacing: .15px\n}\n\n.reference-caption {\n    border-top: 1px solid #edf0e9;\n    padding-top: 12px;\n    margin: 0;\n    font-size: 10px;\n    color: #8e988f\n}\n\n.panel-heading {\n    padding: 20px 23px 17px\n}\n\n.text-button {\n    color: var(--teal);\n    background: transparent;\n    border: 0;\n    font-size: 10px;\n    font-weight: 650;\n    padding: 6px 0;\n    white-space: nowrap\n}\n\n.table-scroll {\n    overflow: auto\n}\n\ntable {\n    border-collapse: collapse;\n    width: 100%;\n    text-align: right;\n    white-space: nowrap;\n    font-size: 11px;\n    font-variant-numeric: tabular-nums\n}\n\nth {\n    background: #f7f9f4;\n    color: #6b7c72;\n    font-size: 10px;\n    font-weight: 600;\n    padding: 11px 12px;\n    line-height: 1.3;\n    border-top: 1px solid var(--line);\n    border-bottom: 1px solid var(--line)\n}\n\nth small {\n    display: block;\n    color: #9aa69a;\n    font-size: 8px;\n    font-weight: 400;\n    margin-top: 4px\n}\n\nth:first-child,td:first-child {\n    text-align: left;\n    padding-left: 23px\n}\n\nth:last-child,td:last-child {\n    padding-right: 23px\n}\n\ntd {\n    padding: 14px 12px;\n    border-bottom: 1px solid #edf0e9;\n    color: #506963\n}\n\ntbody tr:last-child td {\n    border-bottom: 0\n}\n\ntd:first-child {\n    font-weight: 650;\n    color: #254f4b\n}\n\n.selected-row {\n    background: #f0f8f4\n}\n\n.model-color {\n    display: inline-block;\n    height: 6px;\n    width: 6px;\n    border-radius: 50%;\n    margin-right: 8px;\n    vertical-align: middle\n}\n\n.table-note {\n    font-size: 9px;\n    color: #8a978f;\n    line-height: 1.6;\n    margin: 0;\n    padding: 12px 23px 15px;\n    border-top: 1px solid #edf0e9\n}\n\n.plot-tabs {\n    display: flex;\n    gap: 2px;\n    margin: 0 23px;\n    border-bottom: 1px solid var(--line);\n    overflow-x: auto\n}\n\n.plot-tabs button {\n    border: 0;\n    border-bottom: 2px solid transparent;\n    background: transparent;\n    color: #8a988e;\n    padding: 10px 12px;\n    font-size: 10px;\n    white-space: nowrap\n}\n\n.plot-tabs button:first-child {\n    padding-left: 0;\n    padding-right: 14px\n}\n\n.plot-tabs button[aria-selected=true] {\n    color: var(--teal);\n    font-weight: 650;\n    border-bottom-color: var(--teal)\n}\n\n.plot-tools {\n    display: flex;\n    justify-content: space-between;\n    align-items: center;\n    gap: 8px;\n    margin: 16px 23px 6px;\n    font-size: 9px;\n    color: #8c998f\n}\n\n.sweep-controls {\n    display: flex;\n    align-items: center;\n    gap: 6px\n}\n\n.sweep-controls label {\n    font-size: 9px;\n    margin: 0;\n    color: #8c998f;\n    font-weight: 400\n}\n\n.sweep-controls input {\n    height: 26px;\n    width: 58px;\n    border-radius: 4px;\n    padding: 3px 5px;\n    font-size: 9px\n}\n\n.sweep-controls button {\n    font-size: 9px;\n    border: 1px solid #d9e2d7;\n    border-radius: 4px;\n    height: 26px;\n    background: #f7faf3;\n    color: var(--teal);\n    padding: 0 8px\n}\n\n#plot-area {\n    padding: 0 9px\n}\n\n#plot-area svg {\n    display: block;\n    width: 100%;\n    height: auto;\n    max-height: 400px;\n    font-family: Inter,"Segoe UI",Arial,sans-serif;\n    overflow: visible\n}\n\n.plot-legend {\n    display: flex;\n    justify-content: center;\n    align-items: center;\n    gap: 20px;\n    font-size: 10px;\n    color: #6b7b72;\n    margin: 4px 0 15px\n}\n\n.legend-line {\n    display: inline-block;\n    width: 18px;\n    height: 2px;\n    margin: 0 7px 3px 0\n}\n\n.plot-bottom {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    gap: 10px;\n    padding: 10px 23px 13px;\n    border-top: 1px solid #edf0e9\n}\n\n.plot-bottom p {\n    font-size: 9px;\n    color: #8a978f;\n    line-height: 1.6;\n    margin: 0\n}\n\n.small-tag {\n    font-size: 10px;\n    font-weight: 650;\n    color: var(--teal);\n    background: var(--teal-pale);\n    border-radius: 4px;\n    padding: 3px 7px\n}\n\n.roots-panel th,.roots-panel td {\n    font-size: 10px\n}\n\n.selected-dot {\n    display: inline-block;\n    border: 1px solid #b4d9cb;\n    color: var(--teal);\n    background: #e8f5ed;\n    border-radius: 4px;\n    font-size: 8px;\n    font-weight: 500;\n    padding: 2px 4px;\n    margin-left: 7px\n}\n\n.status-bad {\n    color: #ad6b4a\n}\n\n.validation-content {\n    padding: 0 23px 20px;\n    color: #86958b;\n    font-size: 11px\n}\n\n.check-summary {\n    display: flex;\n    align-items: center;\n    gap: 10px;\n    margin-bottom: 9px;\n    color: #33745b;\n    font-weight: 600\n}\n\n.check-summary.fail {\n    color: #a45043\n}\n\n.checks-list {\n    margin: 8px 0 0;\n    padding: 0;\n    list-style: none;\n    display: grid;\n    grid-template-columns: 1fr 1fr;\n    gap: 8px;\n    font-size: 10px\n}\n\n.checks-list li {\n    display: flex;\n    gap: 7px\n}\n\n.check-icon {\n    color: var(--teal)\n}\n\n.checks-list .failed .check-icon {\n    color: #ad5743\n}\n\n.message {\n    font-size: 11px;\n    padding: 12px 16px;\n    border-radius: 9px;\n    border: 1px solid #ead9b6;\n    background: #fffaed;\n    color: #94743c;\n    line-height: 1.6\n}\n\n.message.error {\n    border-color: #efd3cb;\n    background: #fff3ef;\n    color: #ac5140\n}\n\n.message+.message {\n    margin-top: 8px\n}\n\n[hidden] {\n    display: none!important\n}\n\n.empty {\n    text-align: center!important;\n    color: #97a195;\n    padding: 25px!important;\n    font-size: 11px;\n    font-weight: 400!important\n}\n\n.plot-empty {\n    height: 260px;\n    display: flex;\n    align-items: center;\n    justify-content: center\n}\n\n.stale .state-panel,.stale .model-panel,.stale .roots-panel {\n    opacity: .6\n}\n\n.methods {\n    display: grid;\n    grid-template-columns: 300px 1fr;\n    gap: 23px;\n    border-top: 1px solid #dce2d7;\n    margin-top: 42px;\n    padding: 32px 0 20px\n}\n\n.methods>div:first-child {\n    padding-right: 24px\n}\n\n.methods h2 {\n    font-size: 23px;\n    font-weight: 550;\n    letter-spacing: -.7px;\n    line-height: 1.3;\n    margin-bottom: 12px\n}\n\n.methods p {\n    font-size: 11px;\n    color: #849081;\n    line-height: 1.8;\n    margin-top: 0\n}\n\n.method-grid {\n    display: grid;\n    grid-template-columns: 1fr 1fr;\n    gap: 21px 30px\n}\n\n.method-grid h3 {\n    font-size: 11px;\n    font-weight: 650;\n    color: #4c6e5f;\n    margin: 0 0 9px\n}\n\n.method-grid p {\n    font-size: 10px;\n    margin-bottom: 0\n}\n\n.method-grid .formula {\n    font-size: 10px;\n    font-family: Georgia,serif;\n    color: #5f7b68;\n    line-height: 1.7;\n    margin: 0 0 6px\n}\n\n.method-grid a {\n    color: #648776;\n    text-underline-offset: 3px\n}\n\nfooter {\n    border-top: 1px solid #dce2d7;\n    max-width: 1472px;\n    margin: 0 auto;\n    padding: 21px 0 30px;\n    display: flex;\n    align-items: center;\n    gap: 25px;\n    font-size: 9px;\n    color: #97a18e\n}\n\nfooter strong {\n    font-size: 13px;\n    font-weight: 650;\n    color: #4d6b56;\n    letter-spacing: -.3px\n}\n\nfooter>span:last-child {\n    margin-left: auto\n}\n\n.loading #calculate {\n    opacity: .65\n}\n\n.loading #calculate span {\n    display: none\n}\n\n.validation-content details summary {\n    cursor: pointer\n}\n\n.validation-content pre {\n    white-space: pre-wrap;\n    overflow-wrap: anywhere;\n    font-size: 9px;\n    background: #f5f7f2;\n    padding: 10px;\n    border-radius: 5px\n}\n\n.root-selected {\n    background: #f0f8f4\n}\n\n@media(min-width: 1500px) {\n    .layout {\n        grid-template-columns:320px minmax(0,1fr)\n    }\n\n    .methods {\n        grid-template-columns: 320px 1fr\n    }\n\n    .workspace {\n        padding-top: 39px\n    }\n\n    .intro {\n        padding-bottom: 34px\n    }\n\n    .metric strong {\n        font-size: 34px\n    }\n\n    .metric>span {\n        font-size: 11px\n    }\n\n    table {\n        font-size: 12px\n    }\n\n    th {\n        font-size: 11px\n    }\n}\n\n@media(max-width: 1100px) {\n    .workspace {\n        padding:24px\n    }\n\n    .topbar {\n        padding: 0 24px\n    }\n\n    .layout {\n        grid-template-columns: 260px minmax(0,1fr);\n        gap: 17px\n    }\n\n    .controls {\n        padding: 19px\n    }\n\n    .presets {\n        margin-left: -19px;\n        margin-right: -19px;\n        padding-left: 19px;\n        padding-right: 19px\n    }\n\n    .metric strong {\n        font-size: 23px\n    }\n\n    .metric {\n        padding-left: 12px\n    }\n\n    .metric>span {\n        font-size: 9px\n    }\n\n    .state-panel {\n        padding: 19px\n    }\n\n    .panel-heading {\n        padding: 19px\n    }\n\n    .plot-tools {\n        align-items: start;\n        flex-direction: column\n    }\n\n    .methods {\n        grid-template-columns: 260px 1fr;\n        gap: 17px\n    }\n\n    footer {\n        margin: 0 24px\n    }\n\n    .edition {\n        display: none\n    }\n\n    .plot-tabs {\n        margin: 0 19px\n    }\n\n    .plot-tabs button {\n        font-size: 9px;\n        padding-left: 10px;\n        padding-right: 10px\n    }\n}\n\n@media(max-width: 850px) {\n    .layout {\n        grid-template-columns:1fr\n    }\n\n    .controls {\n        position: static\n    }\n\n    .controls form {\n        display: grid;\n        grid-template-columns: 1fr 1fr;\n        gap: 0 22px\n    }\n\n    .controls form>label:first-of-type,.controls form>select:first-of-type,.fluid-meta {\n        grid-column: 1\n    }\n\n    .controls form>.input-pair {\n        grid-column: 2;\n        grid-row: 1 / span 3;\n        align-self: end\n    }\n\n    .controls form>.input-pair label {\n        margin-top: 0\n    }\n\n    .controls form>label[for=phase] {\n        grid-row: 4\n    }\n\n    .controls form>#phase {\n        grid-row: 5\n    }\n\n    .controls form>.field-note:not(.update-note) {\n        grid-row: 6\n    }\n\n    .reference-options {\n        grid-column: 2;\n        grid-row: 4 / span 3;\n        margin: 17px 0 0;\n        padding: 0;\n        border: 0;\n        align-self: start\n    }\n\n    .controls form>.primary {\n        grid-column: 1;\n        margin-top: 18px\n    }\n\n    .controls form>.update-note {\n        grid-column: 2;\n        margin-top: 24px;\n        text-align: left\n    }\n\n    .controls .section-heading {\n        margin-bottom: 7px\n    }\n\n    .presets {\n        display: flex;\n        align-items: center;\n        gap: 20px;\n        margin-top: 16px;\n        padding-top: 17px\n    }\n\n    .presets .eyebrow {\n        white-space: nowrap;\n        margin: 0\n    }\n\n    .presets button {\n        padding: 0;\n        border: 0\n    }\n\n    .sidebar-note {\n        display: none\n    }\n\n    .metrics {\n        margin-top: 23px\n    }\n\n    .metric strong {\n        font-size: 29px\n    }\n\n    .metric>span {\n        font-size: 10px\n    }\n\n    .plot-tools {\n        flex-direction: row;\n        align-items: center\n    }\n\n    .methods {\n        grid-template-columns: 1fr\n    }\n\n    .methods>div:first-child {\n        max-width: 500px\n    }\n\n    .methods h2 {\n        font-size: 23px\n    }\n\n    .method-grid {\n        gap: 22px\n    }\n\n    .header-right {\n        gap: 18px\n    }\n\n    .topbar {\n        height: 76px\n    }\n\n    h1 {\n        font-size: 29px\n    }\n\n    .checks-list {\n        grid-template-columns: 1fr 1fr\n    }\n}\n\n@media(max-width: 560px) {\n    .workspace {\n        padding:20px 14px\n    }\n\n    .topbar {\n        padding: 0 16px\n    }\n\n    .brand {\n        font-size: 20px;\n        gap: 8px\n    }\n\n    .brand-symbol {\n        font-size: 25px;\n        padding: 7px 9px\n    }\n\n    .brand small {\n        font-size: 7px;\n        letter-spacing: 1.2px\n    }\n\n    .header-right .offline {\n        display: none\n    }\n\n    .header-right>a {\n        font-size: 10px\n    }\n\n    .intro {\n        padding: 0 0 23px\n    }\n\n    h1 {\n        font-size: 25px;\n        letter-spacing: -.8px\n    }\n\n    .intro>div>p:last-child {\n        font-size: 11px;\n        max-width: 300px\n    }\n\n    .eyebrow {\n        font-size: 8px;\n        letter-spacing: 1.2px\n    }\n\n    .controls {\n        padding: 18px\n    }\n\n    .controls form {\n        display: block\n    }\n\n    .controls form>.input-pair label {\n        margin-top: 17px\n    }\n\n    .controls .section-heading {\n        margin-bottom: 20px\n    }\n\n    .reference-options {\n        margin: 20px 0;\n        border-top: 1px solid var(--line);\n        padding-top: 16px\n    }\n\n    .controls form>.update-note {\n        text-align: center;\n        margin-top: 8px\n    }\n\n    .presets {\n        margin-left: -18px;\n        margin-right: -18px;\n        gap: 12px;\n        padding-left: 18px;\n        padding-right: 18px;\n        flex-wrap: wrap\n    }\n\n    .presets .eyebrow {\n        flex-basis: 100%\n    }\n\n    .presets button {\n        width: calc(33.33% - 8px);\n        font-size: 9px\n    }\n\n    .presets button small {\n        font-size: 8px\n    }\n\n    .state-panel {\n        padding: 18px 16px\n    }\n\n    .section-heading {\n        gap: 7px\n    }\n\n    h2 {\n        font-size: 13px\n    }\n\n    .step {\n        width: 23px;\n        height: 22px;\n        line-height: 22px;\n        font-size: 9px\n    }\n\n    .focus-label {\n        font-size: 8px;\n        gap: 5px\n    }\n\n    .focus-label select {\n        width: 62px;\n        height: 29px;\n        font-size: 10px\n    }\n\n    .state-caption {\n        font-size: 9px;\n        align-items: flex-start\n    }\n\n    .phase-pill {\n        font-size: 8px\n    }\n\n    .metrics {\n        grid-template-columns: 1fr 1fr;\n        gap: 20px 0;\n        margin: 22px 0 20px\n    }\n\n    .metric {\n        padding-left: 18px\n    }\n\n    .metric:nth-child(3) {\n        padding-left: 0\n    }\n\n    .metric:nth-child(2) {\n        border-right: 0\n    }\n\n    .metric strong {\n        font-size: 29px\n    }\n\n    .metric small {\n        font-size: 9px\n    }\n\n    .reference-caption {\n        font-size: 9px\n    }\n\n    .panel-heading {\n        padding: 18px 16px 14px\n    }\n\n    .text-button {\n        font-size: 9px\n    }\n\n    th:first-child,td:first-child {\n        padding-left: 16px\n    }\n\n    th:last-child,td:last-child {\n        padding-right: 16px\n    }\n\n    .table-note {\n        padding-left: 16px;\n        padding-right: 16px\n    }\n\n    .plot-tabs {\n        margin: 0 16px\n    }\n\n    .plot-tools {\n        flex-direction: column;\n        align-items: flex-start;\n        gap: 12px;\n        margin: 15px 16px 8px\n    }\n\n    .plot-bottom {\n        padding-left: 16px;\n        padding-right: 16px;\n        align-items: flex-start\n    }\n\n    .plot-bottom p {\n        max-width: 220px;\n        font-size: 8px\n    }\n\n    .plot-legend {\n        gap: 15px;\n        font-size: 9px\n    }\n\n    .legend-line {\n        width: 14px;\n        margin-right: 5px\n    }\n\n    .validation-content {\n        padding: 0 16px 18px\n    }\n\n    .checks-list {\n        grid-template-columns: 1fr\n    }\n\n    .methods {\n        margin-top: 30px;\n        padding-top: 26px\n    }\n\n    .method-grid {\n        grid-template-columns: 1fr;\n        gap: 20px\n    }\n\n    .method-grid p {\n        font-size: 11px\n    }\n\n    .methods p {\n        font-size: 11px\n    }\n\n    .methods h2 {\n        font-size: 23px\n    }\n\n    footer {\n        margin: 0 16px;\n        padding-top: 18px;\n        flex-wrap: wrap;\n        gap: 10px;\n        font-size: 8px\n    }\n\n    footer>span:last-child {\n        margin-left: 0;\n        flex-basis: 100%\n    }\n\n    .plot-panel {\n        overflow: hidden\n    }\n\n    #plot-area {\n        padding: 0\n    }\n\n    #plot-area svg {\n        min-height: 230px\n    }\n\n    table {\n        font-size: 10px\n    }\n\n    .roots-panel th,.roots-panel td {\n        font-size: 9px\n    }\n}\n\n@media(prefers-reduced-motion:reduce) {\n    * {\n        scroll-behavior: auto!important;\n        transition: none!important\n    }\n}'

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def validation_checks():
    checks=[]
    def check(name, fn):
        try:
            fn()
            checks.append({"name":name,"passed":True})
        except Exception as exc:
            checks.append({"name":name,"passed":False,"detail":str(exc)})
    def near(a,b,tol):
        if not math.isfinite(a) or abs(a-b)>tol:
            raise AssertionError(f"{a} differs from {b} by more than {tol}")
    def closure():
        for gas in FLUIDS:
            for t,p in ((150,10),(350,40),(700,100)):
                for m in calculate(gas,t,p)["models"]:
                    for r in m["roots"]:
                        near(r["pressure_relative_error"],0,1e-6)
    def gibbs():
        for m in calculate("CO2",350,40)["models"]:
            for r in m["roots"]: near(r["gibbs_identity_error_J_mol"],0,1e-8)
    def ideal_limit():
        for m in calculate("N2",300,1e-5)["models"]:
            near(m["selected"]["Z"],1,1e-6)
            near(m["selected"]["phi"],1,1e-6)
    def reference():
        for m in calculate("CO2",298.15,1,reference="real")["models"]:
            near(m["selected"]["H_J_mol"],0,1e-8)
            near(m["selected"]["S_J_molK"],0,1e-8)
    def cp_integrals():
        for gas,t in (("CH4",1400),("CO2",1300),("N2",550),("O2",750)):
            dt=.001
            a=ideal_properties(gas,t-dt,1); b=ideal_properties(gas,t+dt,1)
            cp=heat_capacity(gas,t)
            near((b["H_J_mol"]-a["H_J_mol"])/(2*dt),cp,1e-5)
            near((b["S_J_molK"]-a["S_J_molK"])/(2*dt),cp/t,1e-7)
    def roots():
        m=next(m for m in calculate("CH4",150,10)["models"] if m["eos"]=="PR")
        assert len(m["roots"])==3
        assert m["roots"][1]["mechanical_stability"]=="unstable"
        near(m["selected"]["ln_phi"],min(r["ln_phi"] for r in m["roots"] if r["mechanical_stability"]!="unstable"),1e-10)
    def unavailable():
        x=calculate("CH4",150,10)
        assert x["ideal"] is None and x["warnings"]
        assert all(m["selected"]["H_J_mol"] is None for m in x["models"])
    def derivatives():
        t=350;dt=.002
        a=calculate("CO2",t-dt,40);b=calculate("CO2",t+dt,40);c=calculate("CO2",t,40)
        for ma,mb,mc in zip(a["models"],b["models"],c["models"]):
            h=-R*t*t*(mb["selected"]["ln_phi"]-ma["selected"]["ln_phi"])/(2*dt)
            near(h,mc["selected"]["H_res_J_mol"],1e-4)
    for name,fn in (("Pressure closure",closure),("Gibbs identity",gibbs),("Ideal-gas limit",ideal_limit),("Real-fluid reference zero",reference),("Piecewise Cp integration",cp_integrals),("Three-root equilibrium selection",roots),("Unsupported Cp range handled",unavailable),("Fugacity temperature derivative",derivatives)):
        check(name,fn)
    return {"checks":checks}

class Handler(BaseHTTPRequestHandler):
    def send_payload(self,body,mime="application/json; charset=utf-8",status=200):
        if not isinstance(body,str):body=json.dumps(body,allow_nan=False)
        encoded=body.encode("utf-8")
        self.send_response(status); self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    def do_GET(self):
        path=self.path.split("?",1)[0]
        if path=="/":self.send_payload(HTML,"text/html; charset=utf-8")
        elif path=="/style.css":self.send_payload(CSS,"text/css; charset=utf-8")
        elif path=="/app.js":self.send_payload(JS,"text/javascript; charset=utf-8")
        elif path=="/api/fluids":self.send_payload({"fluids":fluids()})
        elif path=="/api/validation":self.send_payload(validation_checks())
        elif path=="/favicon.ico":self.send_payload("","image/x-icon",204)
        else:self.send_payload({"error":"Not found"},status=404)
    def do_POST(self):
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=100000:raise ValueError("Invalid request size")
            q=json.loads(self.rfile.read(length))
            if not isinstance(q,dict):raise ValueError("Expected a JSON object")
            allowed=("fluid_id","T_K","P_bar","phase","reference","T_ref_K","P_ref_bar")
            args={k:q[k] for k in allowed if k in q}
            if self.path=="/api/calculate":self.send_payload(calculate(**args))
            elif self.path=="/api/plots":
                P=args.pop("P_bar")
                self.send_payload({"isotherm":isotherm(args["fluid_id"],args["T_K"],P),"sweep":sweep(**args,P_min_bar=q.get("P_min_bar",.1),P_max_bar=q.get("P_max_bar",100),points=q.get("points",85))})
            else:self.send_payload({"error":"Not found"},status=404)
        except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
            self.send_payload({"error":str(exc)},status=400)
    def log_message(self,*args):pass

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Single-file CubicLab")
    parser.add_argument("--port",type=int,default=8770)
    parser.add_argument("--test",action="store_true",help="Run packaged checks and exit")
    options=parser.parse_args()
    if options.test:
        result=validation_checks()
        for c in result["checks"]:print(("PASS" if c["passed"] else "FAIL")+" "+c["name"]+(": "+c.get("detail","") if not c["passed"] else ""))
        sys.exit(0 if all(c["passed"] for c in result["checks"]) else 1)
    print(f"CubicLab: http://127.0.0.1:{options.port} (Ctrl+C to stop)",flush=True)
    ThreadingHTTPServer(("127.0.0.1",options.port),Handler).serve_forever()
