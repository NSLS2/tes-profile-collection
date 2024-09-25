print(f"Loading {__file__!r} ...")

import os


def path_to_hdf5(scan_or_uid):
    for name, doc in db[scan_or_uid].documents():
        if name == "resource":
           file_path = os.path.join(doc['root'], doc['resource_path'])
           print(file_path)
           return file_path


def scan_duration(scan_or_uid):
    hdr = db[scan_or_uid]
    duration = hdr.stop["time"] - hdr.start["time"]
    print(f"The scan uid='{hdr.start['uid']}' / scan_id={hdr.start['scan_id']} took {duration:.3f} seconds")
    return duration
