from abc import ABC, abstractmethod

class BaseHandler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def process_frame(self, frame_data):
        if self._next_handler:
            return self._next_handler.process_frame(frame_data)
        return frame_data

    @abstractmethod
    def post_process(self, video_data):
        if self._next_handler:
            return self._next_handler.post_process(video_data)
        return video_data
