from n_tuples import ConcatTask
import b2luigi as luigi

from events import events_dict

base_path = "/ceph/jeppelt/girlsday"

class EventGenStarterTask(luigi.WrapperTask):
    batch_system = "local"
    def requires(self):
        for name in events_dict.keys():
            yield ConcatTask(
                name = name,
            )

if __name__ == "__main__":
    luigi.process(
        EventGenStarterTask(), workers = 20
    )