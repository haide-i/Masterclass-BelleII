base_path = '/ceph/jeppelt/girlsday'

event_dict = {
    "mumu": ["13", "-13"],
    "pipi": ["211", "-211"],
    "ksks": ["321", "-321"],
    "eepipi": ["11", "-11", "211", "-211"],
    "eek0": ["11", "-11", "310"],
    "eepi0": ["11", "-11", "111"],
    "ee": ["11", "-11"]
}
events_dict = {}
for i in range(5):
    for key, value in event_dict.items():
        events_dict.update({f"{key}{i}": value})