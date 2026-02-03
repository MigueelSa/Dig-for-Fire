import sys, time, threading, itertools

def loading_animation(message: str = "Loading...", delay: float = 0.1):
    """
    Displays a loading animation in the console.
    Returns a function to stop the animation.
    """
    stop_event = threading.Event()

    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if stop_event.is_set():
                break
            sys.stdout.write(f'\r{message} {c}')
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r')

    t = threading.Thread(target=animate)
    t.start()

    # Return a function that stops the spinner
    return stop_event.set