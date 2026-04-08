#Ver  2x
import threading
from time import time
import numpy as np


class FixedSizeList:
    def __init__(self, max_size=50, time_len=5):
        self.max_size = max_size
        self.items = []
        self.timestamps = []
        self.lock = threading.Lock()
        self.time_len = time_len

    def add(self, item, photo=None):
        with self.lock:
            current_time = time()
            self._remove_expired_items(current_time)

            if len(self.items) >= self.max_size:
                self.items.pop(0)
                self.timestamps.pop(0)

            self.items.append(item)
            self.timestamps.append(current_time)

    def rewrite(self, item, nitem, photo=None):
        with self.lock:
            if 0 <= nitem < len(self.items):
                self.items[nitem] = item
                self.timestamps[nitem] = time()

    def get_items(self):
        with self.lock:
            self._remove_expired_items(time())
            return self.items.copy()

    def _remove_expired_items(self, current_time):
        first_valid_index = 0
        while first_valid_index < len(self.timestamps):
            if current_time - self.timestamps[first_valid_index] <= self.time_len:
                break
            first_valid_index += 1

        if first_valid_index > 0:
            del self.items[:first_valid_index]
            del self.timestamps[:first_valid_index]

    def get_TIMEmin(self):
        if not self.timestamps:
            return time()
        return min(self.timestamps)

    def get_EVmin(self, face_locations):
        current_time = time()
        with self.lock:
            self._remove_expired_items(current_time)

            if len(self.items) == 0:
                return 2, -1, [], None

            # face_locations может быть 1D массивом (один эмбеддинг)
            # сравниваем с каждым элементом в списке
            face_locations = np.array(face_locations)
            
            distances = []
            for item in self.items:
                dist = np.linalg.norm(face_locations - np.array(item))
                distances.append(dist)
            
            face_distance_min = min(distances)
            min_index = distances.index(face_distance_min)

            return face_distance_min, min_index, distances, None