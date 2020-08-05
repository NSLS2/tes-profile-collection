def start_detector(detector):
    """Start detector"""
    detector.settings.acquire.put(0)
    detector.settings.num_images.put(16384)  # this is the maximum allowed number of frames on the IOC side
    detector.settings.acquire_time.put(0.05)
    detector.settings.acquire.put(1)


def read_detector(detector):
    """Read detector intensity"""
    return detector.channel1.rois.roi01.value.get()


def stop_detector(detector):
    """Stop detector"""
    detector.settings.acquire.put(0)
