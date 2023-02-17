import time as ttime


def start_detector(detector):
    """Start detector"""
    detector.settings.acquire.put(0)
    detector.settings.num_images.put(16384)  # this is the maximum allowed number of frames on the IOC side
    detector.settings.acquire_time.put(0.05)
    detector.settings.acquire.put(1)


def read_detector(detector):
    """Read detector intensity"""
    return detector.channel1.rois.roi01.value.get()


def watch_function(motors, detector, *args, **kwargs):
    watch_positions = {name: {'position': []} for name in motors}
    watch_intensities = []
    watch_timestamps = []
    for motor_name, field in motors.items():
        for field_name, motor_obj in field.items():
            watch_positions[motor_name][field_name].append(motor_obj.user_readback.get())
    watch_intensities.append(read_detector(detector))
    watch_timestamps.append(ttime.time())
    return watch_positions, watch_intensities, watch_timestamps


def stop_detector(detector):
    """Stop detector"""
    detector.settings.acquire.put(0)
