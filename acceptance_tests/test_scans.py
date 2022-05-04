# Acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/tests.py

def test_xy_fly_scan():
    """
    xy fly scan test.
    """
    print("Starting fly scan test 2")
    uid, = RE(Batch_xy_fly([0]))
    print("xy Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Test is complete")


def test_E_fly_scan():
    """
    e fly scan test.
    """
    print("Starting E_fly_scan test")
    uid, = RE(Batch_E_fly([0]))
    print("E Fly scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Test is complete")


def test_E_step_scan():
    """
    e step scan test.
    """
    print("Starting xanes scan test")
    uid, = RE(Batch_E_step([0]))
    print("E Step scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Test is complete")

test_xy_fly_scan()
test_E_fly_scan()
test_E_step_scan()
