import datetime
import itertools
import pprint
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, cast


@dataclass
class EventStartOrEnd:
    event: 'Event'
    time: datetime.datetime
    is_start: bool

    def __lt__(self, other):
        o = cast(EventStartOrEnd, other)

        if self.event.name != o.event.name:
            return self.event.name < o.event.name

        return self.time < o.time


@dataclass
class Event:
    name: str
    msg_start: str
    msg_end: str

    @staticmethod
    def parse_line(line: str) -> datetime.datetime:
        date, time, *_ = line.split(' ')
        dt = datetime.datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M:%S,%f')
        return dt

    def check(self, line: str) -> Optional[EventStartOrEnd]:
        if self.msg_start in line:
            dt = self.parse_line(line)
            return EventStartOrEnd(event=self, time=dt, is_start=True)

        if self.msg_end in line:
            dt = self.parse_line(line)
            return EventStartOrEnd(event=self, time=dt, is_start=False)

        return None


@dataclass
class EventInstance:
    event: Event
    start: EventStartOrEnd
    end: Optional[EventStartOrEnd]

    def duration(self) -> float:
        if not self.end:
            return -1.0
        return (self.end.time - self.start.time).total_seconds()

    def __repr__(self) -> str:
        return f"Event {self.event.name}: {self.duration()} ({self.start.time} / {self.end.time if self.end else None})"


def make_events(start_and_ends: Iterable[EventStartOrEnd]) -> List[EventInstance]:
    for event_name, group in itertools.groupby(sorted(start_and_ends), key=lambda x: x.event.name):
        starts = []
        for start_or_end in group:
            if start_or_end.is_start:
                starts.append(start_or_end)
                continue

            assert len(starts) > 0

            start = starts.pop()

            yield EventInstance(event=start.event, start=start, end=start_or_end)

        if len(starts) > 0:
            for start in starts:
                yield EventInstance(event=start.event, start=start, end=None)


def select_lines(events: List[Event], lines: Iterable[str]) -> Iterable[EventStartOrEnd]:
    for line in lines:
        for event in events:
            res = event.check(line)
            if res:
                yield res
                break


events = [
    Event("Reader", "Reader starting fit_read", "Reader finished fit_read"),
    Event("LE", "(LE)] fit is started", "(LE)] fit is finished"),
    Event("TE", "(TE)] fit_transform is started", "(TE)] fit_transform is finished"),
    Event("SparkFeaturePipeline", "SparkFeaturePipeline is started", "SparkFeaturePipeline is finished"),
    Event("LinearLBGFS", "Starting LinearLGBFS", "LinearLGBFS is finished"),
    Event("LinearLBGFS single fold", "fit_predict single fold in LinearLBGFS",
          "fit_predict single fold finished in LinearLBGFS"),
    Event("LGBM", "Starting LGBM fit", "Finished LGBM fit")
]


if __name__ == "__main__":
    log = sys.stdin.readlines()

    event_instances = make_events(select_lines(events, log))
    event_instances = sorted(event_instances, key=lambda x: x.start.time)

    print("========================Events==========================")
    pprint.pprint(event_instances)
    print("==================================================")
