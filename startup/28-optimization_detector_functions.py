# start detector
def start_detector(detector):
    detector.settings.acquire.put(0)
    detector.settings.num_images.put(16384)  # this is the maximum allowed number of frames on the IOC side
    detector.settings.acquire_time.put(0.05)
    detector.settings.acquire.put(1)


# read detector
def read_detector(detector):
    return detector.channel1.rois.roi01.value.get()


# stop detector
def stop_detector(detector):
    detector.settings.acquire.put(0)
