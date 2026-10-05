"""Tests of the Python binding. Build first:

    maturin develop --release   (from crates/wavscat-py)
    pytest crates/wavscat-py/tests
"""
import numpy as np
import pytest

import wavscat


def test_every_golden_bit_matches_on_this_machine():
    assert wavscat.verify() == []


def test_numerics_version_is_reported():
    assert wavscat.numerics_version().startswith("0.")


# 30 s of a 3 Hz tap trace at 30 Hz.
N, SR = 900, 30.0
X = np.sin(2 * np.pi * 3.0 * np.arange(N) / SR)


def test_time_scattering_paths_and_shapes():
    op = wavscat.Scattering1d(n=N, J=7, Q=(8, 1), T_sec=4, sr=SR)
    paths = op.paths
    coefs = op.transform(X)
    assert len(coefs) == len(paths)
    assert paths[0]["path"] == "S0" and paths[1]["path"] == "S1_1"
    assert all(c.dtype == np.float64 for c in coefs)
    assert len({c.shape for c in coefs}) == 1  # one time axis with local averaging
    renormed = op.renorm(coefs)
    np.testing.assert_array_equal(renormed[1], coefs[1])  # order 1 untouched


def test_joint_scattering_renorm_removes_amplitude():
    op = wavscat.ScatteringJtfs(n=N, J=7, J_fr=3, Q=(8, 1), T_sec=6, sr=SR)
    order = [p["order"] for p in op.paths]

    def renormed(scale):
        coefs, s1 = op.transform_with_s1(X * scale)
        return op.renorm(coefs, s1, 1e-300)

    a, b = renormed(1.0), renormed(250.0)
    for o, u, v in zip(order, a, b):
        if o == 2:
            np.testing.assert_allclose(u, v, rtol=1e-9)
    assert any(p["spin"] == 1 for p in op.paths) and any(p["spin"] == -1 for p in op.paths)


def test_feature_helpers():
    v = np.array([3.0, 1, 4, 1, 5, 9])
    assert wavscat.summarise(v, "max") == 9
    assert wavscat.summarise(v, "median") == 3.5
    assert wavscat.eps_quantile(np.array([10.0, 3, 1, 4, 2]), 0.3) == pytest.approx(2.2, abs=1e-15)
    lc = wavscat.log_compress(np.array([-2.0, 0.0, 2.0]), 1.0)
    assert lc[1] == 0 and lc[0] == -lc[2]


def test_bad_arguments_fail_loudly():
    with pytest.raises(TypeError):
        wavscat.Scattering1d(n=N, J=7, Qq=8)
    with pytest.raises(ValueError, match="exceeds the signal length"):
        wavscat.Scattering1d(n=N, J=12)
    with pytest.raises(ValueError, match="global"):
        wavscat.Scattering1d(n=N, J=7, T="local")
    op = wavscat.Scattering1d(n=N, J=7)
    with pytest.raises(ValueError, match="length 900"):
        op.transform(np.zeros(10))


def capture(session, video, clock_seed=41):
    for t in wavscat.frame_times(30.0, 30.0, 0.004, 0.05, clock_seed):
        session.push_frame(video.render(t), int(round(t * 1e6)))


def test_a_clean_tapping_trial_is_accepted_with_features():
    session = wavscat.TappingSession()
    capture(session, wavscat.SyntheticVideo(iti_sd=0.01))
    r = session.finish()
    assert r["accepted"], r["reasons"]
    assert abs(r["qc"]["f0_hz"] - 3.0) < 0.06
    f = r["features"]
    assert len(f["names"]) == len(f["values"]) and f["names"][0] == "f0_hz"
    assert any(n.startswith("jtfs_J2") for n in f["names"])
    assert abs(np.mean(r["itis"]) - 1 / 3) < 0.003


def test_a_second_rhythmic_object_is_rejected():
    session = wavscat.TappingSession()
    capture(session, wavscat.SyntheticVideo(iti_sd=0.01, distractor_hz=4.5))
    r = session.finish()
    assert not r["accepted"] and r["features"] is None
    assert any("another rhythmic movement" in s for s in r["reasons"])


def test_rgba_frames_match_converted_luma():
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, size=(120, 160, 4), dtype=np.uint8)
    r, g, b = (rgba[..., i].astype(np.uint32) for i in range(3))
    luma = ((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8)
    a, c = wavscat.TappingSession(capacity=4), wavscat.TappingSession(capacity=4)
    for k in range(3):
        a.push_rgba(rgba, 1000 + k * 33_333)
        c.push_frame(luma, 1000 + k * 33_333)
    assert a.frames == c.frames == 3
    with pytest.raises(ValueError, match="does not follow"):
        a.push_rgba(rgba, 1000)
