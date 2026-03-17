def log_duration(logger, label, seconds, status=None):
    message = f"[TIME] {label}: {float(seconds):.3f}s"
    if status:
        message = f"{message} ({status})"
    logger.info(message)


def log_training_duration(logger, train_start_time, now):
    log_duration(logger, "end_to_end", float(now) - float(train_start_time))


def log_skipped_duration(logger, label, reason="skipped"):
    logger.info(f"[TIME] {label}: skipped ({reason})")
