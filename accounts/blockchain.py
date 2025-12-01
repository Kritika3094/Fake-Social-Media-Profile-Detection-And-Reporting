import hashlib
import json
import requests

class Blockchain:
    def __init__(self):
        self.chain = []

    def create_block(self, report_data):
        """Hashes the report and adds it to the blockchain."""
        block = {
            'index': len(self.chain) + 1,
            'data': report_data,
            'previous_hash': self.hash(self.chain[-1]) if self.chain else "0"
        }
        self.chain.append(block)
        return self.hash(block)

    @staticmethod
    def hash(block):
        """Creates a SHA-256 hash of a block."""
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

blockchain = Blockchain()
