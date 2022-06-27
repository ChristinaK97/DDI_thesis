import os
import re
import time
import psutil
from io import StringIO

from neo4j import GraphDatabase
from config import NEO4J_PATH, NEO4J_USER, NEO4J_PASSWORD, NEO4J_TRAIN_DB, NEO4J_TEST_DB

NEO4J_BIN_PATH = NEO4J_PATH + 'bin'
NEO4J_IMPORT_PATH = NEO4J_PATH + 'import/'
NEO4J_CONFIG_FILE = NEO4J_PATH + 'conf/neo4j.conf'


class Connect_Neo4j:


    def __init__(self, train):
        """
        :param train: True αν θα προσπελάσει τη βάση με τον γράφο του training set
                      False για το test set
        """
        self.neo4j_restart = False
        self.train = train
        self.chosen_db = self._get_chosen_db()
        self.default_db = self._get_default_db()            ; print('Default:', self.default_db, 'To:', self.chosen_db)
        self.driver, self.session = self._start_session()


    def establish_connection(self):
        """
        Εκκινεί το Neo4j μέσω command prompt, συνδέοντας με τη ζητούμενη βάση
        :return: driver και session ώστε να μπορούν να εκτελεστούν queries.
        """
        """
        1. Αν η βάση όπου βρίσκεται ο γράφος προς επεξεργασία δεν συμφωνεί με
           την προεπιλεγμένη βάση με την οποία εκκινείται (ή τρέχει τώρα) το neo4j :
           2. Αν υπάρχει ενεργή σύνδεση με το neo4j, το τερματίζει
           3. Αλλάζει την προεπιλεγμένη βάση με την επιθυμητή
        4. Εκκινεί σύνδεση με την επιλεγμένη βάση
        """
        if self.chosen_db != self.default_db:  # 1
            if self._check_if_running():       # 2
                self._stop_neo4j()
            self._change_default_db()          # 3

        return self._start_neo4j()             # 4


    def _start_session(self):
        """
        :return: Δημιουργεί μία σύνδεση με τη βάση, σύμφωνα με τα credentials
        από το αρχείο other/config.py
        """
        driver = GraphDatabase.driver(uri="bolt://localhost:7687",
                                      auth=(NEO4J_USER, NEO4J_PASSWORD))
        session = driver.session()
        return driver, session


    def _stop_neo4j(self):
        """
        Αποσυνδέεται από τη βάση και τερματίζει τη λειτουργία του neo4j
        """
        print('stop neo4j')
        self.neo4j_restart = True
        self.session.close()
        self.driver.close()

        for process in ['powershell.exe', 'cmd.exe', 'java.exe']:
            for proc in psutil.process_iter():
                if proc.name() == process:
                    print(proc.name(), proc.pid)
                    proc.kill()


    def _start_neo4j(self):
        """
        :return: driver, session (ενεργή σύνδεση)

        1. Όσο δεν έχει καταφέρει να συνδεθεί με την βάση :
           2. Αν δεν έχει εκτελέσει τις εντολές εκκίνησης του cmd ή απαιτείται επανεκκίνηση της βάσης :
              3. Εκκινεί το neo4j μέσω cmd
              4. Δηλώνει ότι έχει εκτελεστεί η εκκίνηση
           5. Ξεκινά σύνδεση με τη βάση
           6. Αν έχει καταφέρει να συνδεθεί, το δηλώνει για να φύγει από το while (επιτυχής σύνδεση)
           Αλλιώς :
               7. Αν δεν έχει τρέξει την εκκίνηση προηγουμένως, δηλώνει ότι πρέπει να την τρέξει στο επόμενο loop
               8. Περιμένει το neo4j να ξεκινήσει
        """
        print('Waiting for Neo4j...')
        not_connected = True
        neo4j_running = True
        neo4j_started = False

        while not_connected:  # 1

            if not neo4j_running or self.neo4j_restart:  # 2
                os.chdir(NEO4J_BIN_PATH)  # 3
                os.system("start cmd /K neo4j console")
                neo4j_started = True  # 4
                neo4j_running = True
                self.neo4j_restart = False

            self.driver, self.session = self._start_session()  # 5

            if self._check_if_running():  # 6
                not_connected = False
            else:
                not_connected = True
                if not neo4j_started:  # 7
                    neo4j_running = False
                time.sleep(20)  # 8

        return self.driver, self.session



    def _check_if_running(self):
        """
        :return: Εκτελεί ένα απλό query στη βάση. Αν προκύψει exception
        σημαίνει ότι δεν υπάρχει ενεργή σύνδεση με τη βάση και
        επιστρέφει False.
        """
        try:
            self.session.run('CALL db.indexes()')
            return True
        except Exception:
            return False


    def _get_chosen_db(self):
        """
        :return: Το όνομα της βάσης (train ή test), όπου είναι ο αντίστοιχος
        γράφος που επεξεργάζεται τη δεδομένη στιγμή.
        """
        return NEO4J_TRAIN_DB if self.train else NEO4J_TEST_DB


    def _get_default_db(self):
        """
        :return: Το όνομα της προεπιλεγμένης βάσης του neo4j, με την
        οποία θα γίνει σύνδεση κατά την εκκίνηση του.
        """
        with open(NEO4J_CONFIG_FILE, "r") as f:
            for line in f:
                if line.startswith('dbms.default_database'):
                    _, default_db = re.split(' |=', line.strip())
                    break
        return default_db


    def _change_default_db(self):
        """
        Αλλάζει το όνομα της προεπιλεγμένης βάσης στην τρέχουσα επιλεγμένη
        (chosen_db), ώστε το neo4j να ξεκινήσει με αυτήν ως την ενεργή βάση.
        """
        print('change default')
        config_data = StringIO()
        with open(NEO4J_CONFIG_FILE, "r") as f:
            for line in f:
                if line.startswith('dbms.default_database'):
                    line = 'dbms.default_database=' + self.chosen_db + '\n'
                config_data.write(line)
        open(NEO4J_CONFIG_FILE, "w").write(config_data.getvalue())

