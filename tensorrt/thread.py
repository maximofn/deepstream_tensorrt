import threading
import sys
import select

class InputThread(threading.Thread):
    data = None

    def run(self):
        # Configuramos el descriptor de archivo de la entrada de la terminal como stdin.
        input_fd = sys.stdin.fileno()

        # Usamos el método select.select() para ver si hay algún dato disponible
        # para leer en la entrada de la terminal.
        read_fds, _, _ = select.select([input_fd], [], [])

        # Si read_fds contiene el descriptor de archivo de la entrada de la terminal,
        # significa que hay algún dato disponible para leer. En ese caso, leemos el
        # dato utilizando el método sys.stdin.readline().
        if input_fd in read_fds:
            self.data = sys.stdin.readline()
    
    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = None