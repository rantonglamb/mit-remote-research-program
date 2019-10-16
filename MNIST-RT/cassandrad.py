import logging
import datetime
from cassandra.cluster import Cluster

KEYSPACE = "info"

def testkeySpace(log):
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()
    log.info("Checking keyspace...")
    result = False
    try:
        rows = session.execute("""
            SELECT * from system_schema.keyspaces
            WHERE keyspace_name = '%s'
        """ % KEYSPACE)
        if rows:
            log.info("Keyspace %s exists" % KEYSPACE)
            result = True
        else:
            log.error("No keyspace")
            result = False
    except Exception as e:
        log.error("No keyspace")
        log.error(e)
        result = False
    return result

def init_logging():
    log = logging.getLogger()
    log.setLevel('INFO')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(handler)
    return log

def insert_value(timestamp, filename, predict, log):
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()
    if not testkeySpace(log):
        return
    try:
        session.set_keyspace(KEYSPACE)
        session.execute(
            """
            INSERT INTO mnist_record(timestamp, filename, predict)
            VALUES (%s, %s, %s)
            """,
            (timestamp, filename, predict)
        )
        log.info("Data written")
    except Exception as e:
        log.error("Unable to write data")
        log.error(e)

def createKeySpace(log):
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()
    log.info("Checking keyspace...")
    try:
        session.execute("""
           CREATE KEYSPACE %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
           """ % KEYSPACE)
        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        log.info("creating table...")
        session.execute("""
           CREATE TABLE mnist_record (
               timestamp timestamp,
               filename text,
               predict int,
               PRIMARY KEY (timestamp)
           )
           """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)



def write_cassandra(filename, predict):
    log = init_logging()
    if not testkeySpace(log):
        createKeySpace(log)
    insert_value(int(datetime.datetime.now().timestamp()*1000), filename, predict, log)