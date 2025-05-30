QUICK_MOTION_MSG = {
    "status": "Too quick, please slow down your motion!",
    "emoji": "U+231B"
}

FOLLOWUP_MSG = {
    "status": "Please place your face inside a red area"
}

SPAMMING_MSG = {
    "status": "Face existed! Please come back later",
    "emoji": "U+26D4"
}

DETECTION_START_MSG = lambda wait_time: {
    "status": f"Stop your motion for {int(wait_time)} second",
    "emoji": "U+1F31F"
}

FACE_NOT_FOUND_MSG = {
    "status": "Face Not Found/ Unregistered",
    "emoji": "U+274C"
}

FACE_SINGED_MSG = lambda user_name: {
    "status": f"User: {user_name} has checked in done!",
    "emoji": "U+2705"
}