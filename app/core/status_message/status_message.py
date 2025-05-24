QUICK_MOTION_MSG = {
    "status": "Too quick, please slow down your motion!",
    "emoji": "🏃"
}

FOLLOWUP_MSG = {
    "status": "Please place your face inside a red area"
}

DETECTION_START_MSG = lambda wait_time: {
    "status": f"Stop your motion for {int(wait_time)} second",
    "emoji": "🙂‍↔️"
}