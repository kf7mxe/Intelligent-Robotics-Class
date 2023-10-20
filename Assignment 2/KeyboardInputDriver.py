from sshkeyboard import listen_keyboard

class KeyboardInput:

    def __init__(self):
        self.enabled = False
        self.listeners = {}

    def _on_press(self, key):
        if key in self.listeners:
            self.listeners[key](True)

    def _on_release(self, key):
        if key in self.listeners:
            self.listeners[key](False)

    def add_listener(self, key, function):
        self.listeners[key] = function

    def enable(self, isOn = True):
        self.listenter = listen_keyboard(on_press=self._on_press, on_release=self._on_release, sequential=False)
