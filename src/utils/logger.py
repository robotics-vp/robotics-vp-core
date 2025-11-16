import csv, os

class CsvLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.header_written = False

    def log(self, **kwargs):
        write_header = (not self.header_written) and (not os.path.exists(self.path))
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(kwargs.keys()))
            if write_header: w.writeheader()
            w.writerow(kwargs)
        self.header_written = True
