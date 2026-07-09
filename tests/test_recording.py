"""Round-trip tests for the in-game EOG recording writer (eog-v3 schema)."""

import numpy as np
import pytest

from brainpong.recording import save_eog_recording, SIGNAL_UNIT


def _save(tmp_path, subject="german", stamp="20260709-171500", slot="P1",
          version="1.3", eeg=None, events=((), ())):
    if eeg is None:
        eeg = np.vstack([np.linspace(-1e-3, 1e-3, 500),      # ch_L (volts)
                         np.linspace(1e-3, -1e-3, 500)])     # ch_R
    ev_s, ev_l = events
    return save_eog_recording(
        tmp_path, subject, eeg, unix_start=1_800_000_000.5, sr=250,
        gain=24, board=f"CERELOG_X8 unit:v{version}",
        montage="canthi differential + bias ear clip, no ground",
        notes="test", tags=["in-game", "2-player"], ch_L=0, ch_R=1,
        n_players=2, board_version=version, serial_port="/dev/cu.usbserial-1110",
        player_slot=slot, sigma_thr=4.0, hpf_hz=0.1, lpf_hz=50.0,
        glance_window_s=0.5, event_samples=ev_s, event_labels=ev_l, stamp=stamp)


def test_roundtrip_fields(tmp_path):
    path = _save(tmp_path)
    assert path.name == "20260709-171500-german.npz"
    d = np.load(path, allow_pickle=True)
    # raw signal: exactly the 2 EOG rows, unfiltered, volts
    assert d['eeg'].shape == (2, 500)
    assert int(d['eog_ch_L'][0]) == 0 and int(d['eog_ch_R'][0]) == 1
    assert str(d['signal_unit'][0]) == SIGNAL_UNIT == "volts"
    assert str(d['protocol_version'][0]) == "eog-v3"
    assert str(d['subject_id'][0]) == "german"
    assert float(d['unix_start'][0]) == pytest.approx(1_800_000_000.5)
    # game-context fields
    assert int(d['n_players'][0]) == 2
    assert str(d['board_version'][0]) == "1.3"
    assert str(d['serial_port'][0]) == "/dev/cu.usbserial-1110"
    assert str(d['player_slot'][0]) == "P1"
    assert float(d['sigma_thr'][0]) == 4.0
    assert float(d['hpf_hz'][0]) == 0.1
    assert float(d['lpf_hz'][0]) == 50.0
    assert float(d['glance_window_s'][0]) == 0.5


def test_event_markers(tmp_path):
    path = _save(tmp_path, events=([10, 130], ["calib_start", "play_start"]))
    d = np.load(path, allow_pickle=True)
    assert list(d['event_samples']) == [10, 130]
    assert [str(x) for x in d['event_labels']] == ["calib_start", "play_start"]


def test_two_players_same_stamp_distinct_files(tmp_path):
    p1 = _save(tmp_path, subject="german", slot="P1", version="1.2")
    p2 = _save(tmp_path, subject="john", slot="P2", version="1.3")
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p1.name == "20260709-171500-german.npz"
    assert p2.name == "20260709-171500-john.npz"


def test_same_subject_collision_gets_slot_suffix(tmp_path):
    a = _save(tmp_path, subject="P1", slot="P1")
    b = _save(tmp_path, subject="P1", slot="P2")   # same ts+subject → suffixed
    assert a.name == "20260709-171500-P1.npz"
    assert b.name == "20260709-171500-P1-P2.npz"


def test_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        _save(tmp_path, eeg=np.zeros((8, 100)))   # must be (2, N)
