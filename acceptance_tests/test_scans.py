# Acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/tests.py

def test_xy_fly_scan():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(Batch_xy_fly([0]))
    print("xy Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_E_step_scan():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting xanes scan test")
    uid, = RE(Batch_E_step([0]))
    print("E Step scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Exporting scan ...")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")

def test_E_fly_scan():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting xanes scan test")
    uid, = RE(Batch_E_fly([0]))
    print("E Fly scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Exporting scan ...")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


test_xy_fly_scan()
test_E_step_scan()
test_E_fly_scan()
