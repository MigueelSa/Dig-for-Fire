import sys, time, threading, itertools

def loading_animation(message: str | list  = "Loading...", delay: float = 0.1):
    """
    Displays a loading animation in the console.
    Returns a function to stop the animation.
    """
    stop_event = threading.Event()

    if isinstance(message, str):
        message = [message]

    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if stop_event.is_set():
                break
            sys.stdout.write(f'\r{message[0]} {c}')
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write('\r' + ' ' * (len(message[0]) + 2) + '\r')

    t = threading.Thread(target=animate)
    t.start()

    # Return a function that stops the spinner
    return stop_event.set

class ProgressTracker:
    def __init__(self, total: int):
        self.current: int = 0
        self.total: int = total
        self.start_time: float = time.time()

    def update(self, current: int):
        self.current = current

    @property
    def ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.current/self.total
    
    @property
    def eta(self) -> float:
        """Estimated seconds remaining"""
        if self.current == 0:
            return -1
        elapsed = time.time() - self.start_time
        remaining = (elapsed / self.current) * (self.total - self.current)
        return remaining