from collections import Counter
from dataclasses import asdict, replace
from datetime import date

import pytest
from factories import ABROAD, RIVAL, snapshot, staff_member
from test_classify import facts
from test_fmsave_source import UID_MAP, coded, fm_staff

from facegen.core.classify import Group, Rules, classify
from facegen.core.traits import sample_sheet
from facegen.core.wardrobe import role_for
from facegen.datasource.fmsave_source import convert_staff
from facegen.datasource.kits import load_kits
from facegen.datasource.photos import PhotoIndex, PhotoScan
from facegen.persistence.jobs import JobFilter, allocate_sheet, claim, load_career, queued_jobs
from facegen.pipeline.sync import run_sync
from facegen.pipeline.worker import Worker


def test_read_only_my_staff_and_league_first_team_managers():
    records = convert_staff(
        [
            fm_staff(1),
            fm_staff(2, club_uid=11, job=coded("MANAGER"), team_slot=0),
            fm_staff(3, club_uid=11, job=coded("ASSISTANT_MANAGER")),
            fm_staff(4, club_uid=12, job=coded("MANAGER")),
            fm_staff(5, club_uid=11, job=coded("MANAGER"), team_slot=2),
            fm_staff(6, club_uid=11, job=coded("MANAGER"), team_slot=None),
        ],
        UID_MAP,
        10,
        date(2026, 11, 13),
        manager_club_ids=frozenset({1011}),
    )
    assert [(r.unique_id, r.club_id) for r in records] == [(1, 1010), (2, 1011)]


@pytest.mark.parametrize("photo", [True, False, None])
@pytest.mark.parametrize("newgen", [True, False])
def test_manager_requires_known_absence_of_any_photo(settings, photo, newgen):
    manager = facts(
        is_staff=True,
        staff_job="manager",
        in_my_league=True,
        photo=photo,
        is_newgen=newgen,
        ca=199,
        pa=200,
    )
    result = classify(manager, settings.groups)
    assert result.group == (Group.C if photo is False else None)
    assert not result.tracked
    assert classify(replace(manager, at_my_club=True), settings.groups).group is Group.A
    assert classify(replace(manager, pin="never"), settings.groups).group is None
    assert classify(manager, settings.groups, Rules(staff=False)).group is None


def test_sync_manager_photo_and_league_transitions(conn, settings):
    snap = snapshot(
        staff=[
            staff_member(301, job="manager", club_id=RIVAL),
            staff_member(302, job="manager", club_id=RIVAL),
            staff_member(303, job="assistant manager", club_id=RIVAL),
            staff_member(304, job="manager", club_id=ABROAD),
        ]
    )
    scan = PhotoScan({"configs": []}, PhotoIndex(frozenset(), frozenset({302})))
    run_sync(conn, snap, settings, save_file="career.fm", photos=scan)
    assert [
        tuple(r)
        for r in conn.execute("SELECT unique_id, group_name FROM jobs WHERE status='queued'")
    ] == [(301, "c")]
    run_sync(conn, snap, settings, save_file="career.fm", photos=scan)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    scan = PhotoScan({"configs": []}, PhotoIndex(frozenset({301}), frozenset({302})))
    run_sync(conn, snap, settings, save_file="career.fm", photos=scan)
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "cancelled"


def test_manager_hair_is_natural_and_changes_no_other_traits():
    counts = Counter()
    for uid in range(5000):
        normal = sample_sheet("career", uid, 1)
        manager = sample_sheet("career", uid, 1, manager=True)
        counts[manager.hair_colour] += 1
        assert asdict(replace(manager, hair_colour=normal.hair_colour)) == asdict(normal)
        assert sample_sheet("career", uid, 1, manager=True) == manager
    assert 0 < counts["pink"] < 100
    assert 0 < counts["blue"] < 100
    assert counts["black"] + counts["brown"] > 3000
    assert role_for("head coach") == role_for("manager") == "manager"


def test_manager_preview_and_worker_use_the_same_sheet_and_role(conn, settings):
    snap = snapshot(staff=[staff_member(301, job="manager", club_id=RIVAL)])
    scan = PhotoScan({"configs": []}, PhotoIndex(frozenset(), frozenset()))
    run_sync(conn, snap, settings, save_file="career.fm", photos=scan)
    career = load_career(conn)
    worker = Worker(
        conn,
        settings,
        load_kits(),
        career,
        renderer_factory=lambda: None,
        tagger_factory=lambda: None,
    )
    preview = worker.preview(JobFilter())[0]
    selected = queued_jobs(conn, career.id, JobFilter())[0]
    job = claim(conn, career.id, selected)
    version, sheet = allocate_sheet(conn, career, job, settings.body, settings.personality.weight)
    assert sheet == preview.sheet
    assert allocate_sheet(conn, career, job, settings.body, settings.personality.weight) == (
        version,
        sheet,
    )
    setup = worker._setup(job, sheet, 1)
    assert setup.wear.role == "manager"
    assert setup.pose in {"crossed", "behind", "hips"}
