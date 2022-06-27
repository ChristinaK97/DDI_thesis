
from other.CONSTANTS import *
import os

from source.database_pcg.n4j_pcg.establish_neo4j_connection import Connect_Neo4j, NEO4J_IMPORT_PATH


class Initialize_Neo4j:

    def __init__(self, train, triples):
        self.driver, self.session = \
            Connect_Neo4j(train).establish_connection()

        self.indexes = [DRUG_CLASS, DATA_NODE, INTERACTION_CLASS,
                        SENT_CLASS, TOKEN_CLASS, OWL_CLASS]
        self.reset_db()
        self.create_database(triples)


    def reset_db(self):
        for index in self.indexes:
            try:
                self.session.run('DROP INDEX ON :' + index + '(key)')
            except Exception:
                print('index doesnt exist')
        self.session.run('MATCH (n) DETACH DELETE n')


    def create_database(self, triples):

        self.create_indexes()
        self.write_class_nodes()
        self.write_nodes_and_relations(triples)
        self.write_RdfType_relation(triples)
        self.transitive_sameAs()
        self.delete_csv()
        self.session.close()
        self.driver.close()


    def create_indexes(self):
        """
        Δημιουργεί καταλόγους για τους διάφορους τύπους κόμβων του γράφου.
        """
        indexes = [DRUG_CLASS, DATA_NODE, INTERACTION_CLASS, SENT_CLASS, TOKEN_CLASS, OWL_CLASS]
        for index in indexes:
            self.session.run(
                f"CREATE INDEX FOR (d:{index}) ON (d.key)"
            )



    def write_class_nodes(self):
        """
        Γράφει τις τριπλέτες που δείχνουν την ιεραρχία των ορισμένων στην οντολογία κλάσεων
        Για κάθε κλάση που έχει οριστεί
        Φτιάξε κόμβους για τις δύο κλάσεις (αν δεν υπάρχουν ήδη) και ένωσε τις με τη σχέση subClassOf
               MERGE (class_name:owl_Class{key:'class_name'})
               MERGE (superClass_name:owl_Class{key:'superClass_name'}
               CREATE (class_name)-[:rdfs_subClassOf]->(superClass_name)
        """
        for class_ in get_classes():
            superClass = get_superClass(class_)  # Η υπερκλάση της

            query = \
                f'''MERGE ({class_}:{OWL_CLASS} {{key:\"{class_}\"}})\n 
                    MERGE ({superClass}:{OWL_CLASS} {{key:\"{superClass}\"}})\n  
                    CREATE ({class_})-[:{SUBCLASS}]->({superClass})
                '''
            print(query,'_____________', sep='\n')
            self.session.run(query)



    def write_nodes_and_relations(self, t):
        """
        Περνάει τα δεδομένα (τριπλέτες t : κόμβους και σχέσεις μεταξύ τους) στην ΒΔ
        1. Για κάθε predicate p (τύπο relation μεταξύ των κόμβων του γραφήματος):
            2. Ανάκτησε το domain και το range του
            3. (domain None -> rdf_type predicate : το χειρίζεται ξεχωριστά η write_RdfType_relation)
            4. Κράτα τις τριπλέτες που έχουν predicate το p και γράψε τις στην ΒΔ
        """
        preds = t[RDF_P].unique()  # 1

        for p in preds:
            domain, range = get_domain_range(p)  # 2
            if domain is None: continue  # 3

            p_triples = t.loc[t[RDF_P] == p]  # 4
            self.write_df(p_triples, p, domain, range)



    def write_RdfType_relation(self, t):
        """
        Γράφει τις τριπλέτες με predicate rdf_type -> Δηλώνουν τον τύπο κάθε κόμβου
        :param t: Το DF με όλες τις τριπλέτες του γράφου

        1. Κράτα τις τριπλέτες που έχουν predicate rdf_type
        2. Κράτα τις τριπλέτες της μορφής (ξεχωριστά)
               <drug_node, rdf_type, drug_type>
               <_:p{A/A}, rdf_type, interaction_type>
               elements : οι κλάσεις των φαρμάκων (object στις τριπλέτες rdf_type)
               domain   : Drug_Class ή Interaction, ανάλογα με τους κόμβους που εξετάζουμε
           και γράψε τις στη ΒΔ.
        3. Ένωσε τους κόμβους των sentences και tokens με τους κόμβους κλάσεων Sentence και Token
           μέσω σχέσης rdf_type
        """
        node_types = t.loc[t[RDF_P] == RDF_TYPE]  # 1
        for elements, domain in [(DRUG_CLASSES, DRUG_CLASS), (INTERACTION_CLASSES, INTERACTION_CLASS)]:  # 2
            df = node_types.loc[node_types[RDF_O].isin(elements)].reset_index(drop=True)
            self.write_df(df, RDF_TYPE, domain, OWL_CLASS)

        for other_cl in [SENT_CLASS, TOKEN_CLASS]:  # 3
            query = f'MATCH (n:{other_cl}), (c:{OWL_CLASS}) ' \
                    f'WHERE c.key=\"{other_cl}\" CREATE (n)-[r:{RDF_TYPE}]->(c)'

            self.session.run(query)



    def transitive_sameAs(self):
        query = f'''
             MATCH p = (a:{DRUG_CLASS})-[:{SAME_AS}]-(b:{DRUG_CLASS})-[:{SAME_AS}]-(c:{DRUG_CLASS})
             WHERE NOT (a)-[:{SAME_AS}]-(c) AND a.key < c.key
             WITH DISTINCT a, c
             CREATE (a)-[r:{SAME_AS}]->(c)
             RETURN count(r)
        '''
        r = -1
        while r != 0:
            r = self.session.run(query)
            key = r.keys()[0]
            r = [dict(i) for i in r][0][key]


    def write_df(self, df, predicate, domain, range_):
        # 17 sec
        """
        Γράφει στην βάση ένα σύνολο τριπλετων που έχουν ίδιο predicate, domain και range.
        Πρακτικά οι τριπλετες αυτές αντιστοιχούν σε ένα συγκεκριμένο predicate, ένα τύπο
        relation μεταξύ των κόμβων του γραφήματος. Μπορεί να είναι και διάφορα predicates
        αρκεί να έχουν ίδιο domain και ίδιο range.
        :param df: DF με τριπλέτες
        :param predicate: ίδιο κατηγόρημα
        :param domain: το πεδίο ορισμού του κατηγορήματος.
                       Ο τύπος των κόμβων subject
        :param range_:  το πεδίο τιμών του κατηγορήματος.
                       Ο τύπος των κόμβων object
        """
        """
        MERGE (subject_node:τύπος/κλάση του κόμβου {key:το όνομα του κόμβου subject
               που διάβασε στη γραμμή row}
        Όμοια για το object
        Ένωσε τους κόμβους subject_node και object_node με τη σχέση που δηλώνεται στο
        πεδίο rdf_predicate στο row στο αρχείο
        Η ενημέρωση της ΒΔ βα γίνει περιοδικά, με μέγεθος batch 5000 commits
        """
        df.to_csv(NEO4J_IMPORT_PATH + predicate + '_pred.csv', sep=',', encoding='utf-8', index=False)

        query = f'''
            CALL apoc.periodic.iterate(
                'CALL apoc.load.csv(\"file:///{predicate}_pred.csv\",{{sep:\",\"}}) yield map as row RETURN row',
                    'MERGE (subject_node:{domain} {{key:row.{RDF_S}}})
                     MERGE (object_node:{range_} {{key:row.{RDF_O}}})
                     WITH subject_node, object_node, row
                     CALL apoc.create.relationship(subject_node, row.{RDF_P}
                        , {{}}, object_node) YIELD rel RETURN rel'
            , {{batchSize:5000, iterateList:true}})
        '''

        self.session.run(query)

        print(domain, predicate, range_, df.shape); print(query, '\n____________')



    def delete_csv(self):
        """
        Διαγράφει τα αρχεία csv που φτιάχτηκαν στον φάκελο του neo4j
        """
        folder = NEO4J_IMPORT_PATH[:-1]
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if not file_path.endswith('_pred.csv'):
                continue
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

# -----------------------------------------------------------------------------------------------

    def write_df2(self, df, predicate, domain, range_):
        # Πιο αργή μέθοδος εγγραφής, που ωστόσο δε δημιουργεί αρχεία
        """
        Γράφει στην βάση ένα σύνολο τριπλετων που έχουν ίδιο predicate, domain και range.
        Πρακτικά οι τριπλετες αυτές αντιστοιχούν σε ένα συγκεκριμένο predicate, ένα τύπο
        relation μεταξύ των κόμβων του γραφήματος. Μπορεί να είναι και διάφορα predicates
        αρκεί να έχουν ίδιο domain και ίδιο range.
        :param df: DF με τριπλέτες
        :param predicate: ίδιο κατηγόρημα
        :param domain: το πεδίο ορισμού του κατηγορήματος.
                       Ο τύπος των κόμβων subject
        :param range_:  το πεδίο τιμών του κατηγορήματος.
                       Ο τύπος των κόμβων object
        """
        """
        MERGE (subject_node:τύπος/κλάση του κόμβου {key:το όνομα του κόμβου subject
               που διάβασε στη γραμμή row}
        Όμοια για το object
        Ένωσε τους κόμβους subject_node και object_node με τη σχέση που δηλώνεται στο
        πεδίο rdf_predicate στο row στο αρχείο
        Η ενημέρωση της ΒΔ βα γίνει περιοδικά, με μέγεθος batch 5000 commits
        """

        batch_size = 5000
        df = df.to_dict('records')

        query = f'''
            CALL apoc.periodic.iterate(
                'unwind $df as row return row',
                    'MERGE (subject_node:{domain} {{key:row.{RDF_S}}})
                     MERGE (object_node:{range_} {{key:row.{RDF_O}}})
                     WITH subject_node, object_node, row
                     CALL apoc.create.relationship(subject_node, row.{RDF_P}
                        , {{}}, object_node) YIELD rel RETURN rel'
            , {{batchSize:{batch_size}, iterateList:true, params: {{df: $df}} }})
        '''

        for count in range(0, len(df), batch_size):
            length = min(len(df), batch_size + count)
            self.session.run(query, df=df[count:length])  # ;print('index : ', count, length)

        print(domain, predicate, range_, len(df)); print(query, '\n____________')



