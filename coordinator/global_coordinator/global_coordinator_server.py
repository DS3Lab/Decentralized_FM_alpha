from datetime import datetime
import argparse
import json
import pycouchdb


class GlobalCoordinatorServer:
    def __init__(self, args):
        server = pycouchdb.Server(args.db_server_address)
        self.db = server.database("global_coordinator")
        self.allocated_task_index = 0
        self.task_meta_key = None
        self.task_meta = None
        self._resume_server_from_pycouchdb()

    def _resume_server_from_pycouchdb(self):
        for entrance in self.db.all():
            if 'meta_start_time' in entrance['doc']:
                self.task_meta_key = entrance['key']
                self.task_meta = entrance['doc']
        if self.task_meta_key is None:
            self.task_meta_key = self.db.save({
                'meta_start_time': str(datetime.now()),
                'task_hash': {}
            })['_id']
            self.task_meta = self.db.get(self.task_meta_key)
            print(self.task_meta)

    def _allocate_task_index(self):
        current_index = self.allocated_task_index
        self.allocated_task_index += 1
        return current_index

    def check_key_value_info(self):
        for entrance in self.db.all():
            print("-----------------------------------------")
            print(entrance)

    def clear_key_value(self):
        keys = [doc['key'] for doc in self.db.all()]
        print(keys)
        for key in keys:
            self.db.delete(key)

def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Server')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    args = parser.parse_args()
    print(vars(args))
    coordinator = GlobalCoordinatorServer(args)
    # coordinator.clear_key_value()
    coordinator.check_key_value_info()


if __name__ == '__main__':
    main()
