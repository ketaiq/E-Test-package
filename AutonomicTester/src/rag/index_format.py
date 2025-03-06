from enum import Enum


class IndexFormat(Enum):
    RAW = "raw" # indexes from raw Java source code in .java files
    JSON = "json" # indexes from JSON files that preprocesses .java files