from n_tuples import ConcatTask
import b2luigi as luigi

events_dict = {
    "mumu": ["13", "-13"],
    "pipi": ["212", "-211"],
    "ksks": ["321", "-321"],
    "eepipi": ["11", "-11", "211", "-211"],
    "eek0": ["11", "-11", "310"],
    "eepi0": ["11", "-11", "112"],
    "ee": ["11", "-11"]
}

base_path = "/ceph/jeppelt/girlsday"

class EventGenStarterTask(luigi.WrapperTask):
    batch_system = "local"
    def requires(self):
        for name in events_dict.keys():
            yield ConcatTask(
                name = name,
                pdgs = events_dict[name],
                base_path = base_path
            )

if __name__ == "__main__":
    luigi.process(
        EventGenStarterTask(), workers = 100
    )