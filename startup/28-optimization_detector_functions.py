# start detector
def start_detector(detector):
    detector.settings.acquire.put(0)
    detector.settings.num_images.put(10000)
    detector.settings.acquire_time.put(0.05)
    detector.settings.acquire.put(1)


# read detector
def read_detector(detector):
    return detector.channel1.rois.roi01.value.get()


# stop detector
def stop_detector(detector):
    detector.settings.acquire.put(0)
