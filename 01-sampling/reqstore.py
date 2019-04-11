from urllib.parse import urljoin
from io import BytesIO
from itertools import chain
from functools import reduce

import requests
import hydro_serving_grpc as hs
from google.protobuf import text_encoding
import urllib


class APIHelper:

    @staticmethod
    def _urljoin(address, name, path):
        return urljoin(address, "/".join([name, path]))

    @staticmethod
    def status(address, name): 
        url = APIHelper._urljoin(address, name, "status")
        return requests.get(url).json()
        
    @staticmethod
    def download(address, name, from_ts, to_ts):
        url = APIHelper._urljoin(address, name, "get")
        r = requests.get(url, stream=True, params={"from": from_ts, "to": to_ts})
        return r.content

    @staticmethod
    def subsample(address, name):
        status = APIHelper.status(address, name)
        # TODO in case if there was only one request
        #      instead of 'FromTo' reqstore returns range { 'Single' }
        try: 
            rng = status['fs']['range']['FromTo']
        except KeyError as ex:
            raise ValueError(f"Application {name} has no available records.") from ex
        from_ts, to_ts = rng['from'], rng['to']

        print(f"Downloading requests from {name} for timestamps: {from_ts} to {to_ts}")
        return APIHelper.download(address, name, from_ts, to_ts)


class BinaryHelper:

    @staticmethod
    def read_int(binary: BytesIO):
        return int.from_bytes(binary.read(4), byteorder='big')

    @staticmethod
    def read_long(binary: BytesIO):
        return int.from_bytes(binary.read(8), byteorder='big')

    @staticmethod
    def read_message(binary: BytesIO, grpc_msg):
        header = binary.read(1) 
        data = b'' if header == 0 else binary.read()
        grpc_msg.ParseFromString(data)
        return grpc_msg
        
    @staticmethod
    def decode_records(data: bytes):
        bio = BytesIO(data)
        size = len(data)
        records = []
        while size > 0:
            length = BinaryHelper.read_int(bio)
            ts = BinaryHelper.read_long(bio)
            body = BytesIO(bio.read(length))
            entries = BinaryHelper.decode_entries(body)
            records.append(TsRecord(ts, entries))
            size = size - length - 4 - 8
        return records
    
    @staticmethod
    def decode_entries(binary: BytesIO):
        minUid = BinaryHelper.read_long(binary)
        count = BinaryHelper.read_int(binary)
        entries = []
        for i in range(count):
            length = BinaryHelper.read_int(binary)
            data = binary.read(length)
            entry = Entry(minUid + i, data)
            entries.append(entry)
        return entries

    @staticmethod
    def decode_request(data: bytes):
        return BinaryHelper.read_message(BytesIO(data), hs.PredictRequest())

    @staticmethod
    def decode_response(data: bytes):
        bio = BytesIO(data)
        offset = BinaryHelper.read_int(bio)
        data = BytesIO(bio.read()) 

        if offset == 2:
            return BinaryHelper.read_message(bio, hs.ExecutionError())
        elif offset == 3:
            return BinaryHelper.read_message(data, hs.PredictResponse())
        raise UnicodeDecodeError

class Entry:
    def __init__(self, uid, data):
        self.uid = uid
        self.binary = data
        self.__request = None
        self.__response = None

    @property
    def request(self):
        if not self.__request:
            self._read_binary()
        return self.__request

    @property
    def response(self):
        if not self.__response:
            self._read_binary()
        return self.__response

    def _read_binary(self):
        bio = BytesIO(self.binary)

        request_size = BinaryHelper.read_int(bio)
        response_size = BinaryHelper.read_int(bio)
        self.__request = BinaryHelper.decode_request(bio.read(request_size))
        self.__response = BinaryHelper.decode_response(bio.read(response_size))

    def __repr__(self):
        return f"Entry(uid={self.uid})"


class TsRecord:
    def __init__(self, ts, entries):
        self.ts = ts
        self.entries = entries

    def __repr__(self):
        return f"Record(ts={self.ts}, entries={self.entries})"


def splice_entries(records: [TsRecord]):
    return list(chain(*[item.entries for item in records]))


if __name__ == "__main__":
    addr = "http://localhost:7265"
    name = "app1stage0"

    binary_data = APIHelper.subsample(addr, name)
    records = BinaryHelper.decode_records(binary_data)

    i = 1
    for record in records:
        for entry in record.entries:
            print(entry.response); i += 1 
            break
        break