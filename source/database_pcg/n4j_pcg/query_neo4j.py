from source.database_pcg.n4j_pcg.establish_neo4j_connection import Connect_Neo4j
from other.CONSTANTS import *
import pandas as pd

class Query_Neo4j:

    def __init__(self, train):
        self.driver, self.session = \
            Connect_Neo4j(train).establish_connection()

    def close(self):
        self.session.close()
        self.driver.close()

    def run_query(self, query):
        print(query, '---------', sep='\n')

        r = self.session.run(query)
        return r.values()


    def graph_representation_triples(self):
        nodes_per_class = {}
        conditions = ['', f'where (node)-[:{SENT_CON_INTER}]->(:{INTERACTION_CLASS})',
                          f'where (node)<-[:{WITH_TOKEN}]-(:{INTERACTION_CLASS})']

        for node_class, condition in zip([INTERACTION_CLASS, SENT_CLASS, TOKEN_CLASS], conditions):
            nodes_per_class[node_class] = [item[0] for item in
                                           self.run_query(self.q_get_nodes(node_class, condition))]

        queries = [
        f'''
            with '{TOKEN_IN_INT}' as {RDF_P}
            match (p:{INTERACTION_CLASS})-[:{WITH_TOKEN}]->(t:{TOKEN_CLASS})
            return t.key as {RDF_S}, {RDF_P}, p.key as {RDF_O}
        ''',
            self.q_by_predicate(SENT_CON_INTER, directed=True)
        ]
        subgraph = []
        for q in queries:
            subgraph += self.run_query(q)
        subgraph = pd.DataFrame(subgraph, columns=RDF_TRIPLE)

        return nodes_per_class, subgraph


    def q_get_nodes(self, node_class, condition=''):
        query = f'''
            match (node:{node_class})
            {condition}
            return node.key
        '''
        return query

    def get_interaction_nodes_labels(self, label_enc):
        query = f'''
            match (p:{INTERACTION_CLASS})-[:{RDF_TYPE}]->(label:{OWL_CLASS})
            return p.key, label.key
        '''
        node_labels = self.run_query(query)
        node_labels = {interaction_node : label_enc.transform([label])
                       for [interaction_node, label] in node_labels}
        return node_labels


    def q_by_predicate(self, predicate, directed=True):

        domain, range = get_domain_range(predicate)
        arrow = '->' if directed else '-'

        query = f'''with \'{predicate}\' as {RDF_P}
                    match (s:{domain})-[:{predicate}]{arrow}(o:{range})
                    return s.key as {RDF_S}, {RDF_P}, o.key as {RDF_O}
                '''
        return query


    def q_rdf_type(self, domain):
        query = f'''with '{RDF_TYPE}' as {RDF_P} 
                    match (s:{domain})-[:{RDF_TYPE}]->(o:{OWL_CLASS}) 
                    return s.key as {RDF_S}, {RDF_P}, o.key as {RDF_O}
                '''
        return query


    def q_ddi_subgraph(self):
        query = f'''
             match (s:{DRUG_CLASS})-[:{INT_FOUND}]->(i:{INTERACTION_CLASS})<-[:{INT_FOUND}]-(o:{DRUG_CLASS}) 
             match (i)-[:{RDF_TYPE}]->(p:{OWL_CLASS}) 
             return distinct s.key as {RDF_S}, p.key as {RDF_P}, o.key as {RDF_O}, i.key as {INTERACTION_CLASS}
             '''
        return query


    def q_multitype_common_name(self):
        query = f'''
                with 'common_name' as {RDF_P} 
                match (d1:{DRUG_CLASS})-[:{ENT_NAME}]->(n:{DATA_NODE}) 
                match (d2:{DRUG_CLASS})-[:{ENT_NAME}]->(n:{DATA_NODE}) 
                where d1.key <> d2.key 
                return d1.key as {RDF_S}, {RDF_P}, d2.key as {RDF_O}
                '''
        return query


    def q_collect_sentences(self):
        query = f'''
                MATCH (s:{SENT_CLASS})-[:{SENT_TEXT}]->(text:{DATA_NODE}) 
                MATCH (s)-[:{SENT_CON_TOKEN}]->(token:{TOKEN_CLASS}) 
                MATCH (token)-[:{START}]->(st:{DATA_NODE}) 
                MATCH (token)-[:{END}]->(e:{DATA_NODE}) 
                MATCH (drug:{DRUG_CLASS})-[:{ENT_FOUND_AS}]->(token) 
                with text.key as text, drug.key as drug, toInteger(st.key) as st, toInteger(e.key) as e, token.key as token,
                s.key as {SENT_CLASS}
                where st < e
                with text, collect({{{DRUG_CLASS}:drug, {START}:st, {END}:e, {TOKEN_CLASS}:token}}) AS tokens, {SENT_CLASS}
                RETURN text AS {SENT_TEXT}, tokens, {SENT_CLASS}
                ORDER BY text
            '''
        return query


    def q_sentences_interactions(self):
        query = f'''
            match (text:{DATA_NODE})<-[:{SENT_TEXT}]-(s:{SENT_CLASS})-[:{SENT_CON_INTER}]->(i:{INTERACTION_CLASS})
            match (d1:{DRUG_CLASS})-[:{INT_FOUND}]->(i)<-[:{INT_FOUND}]-(d2:{DRUG_CLASS})
            with text.key as text, collect({{{E1}:d1.key, {INTERACTION_CLASS}:i.key, {E2}:d2.key}}) as interactions
            return interactions
            order by text
        '''
        return query


    def new_pairs_from_sameAs(self):
        query = f'''
            match (d1:{DRUG_CLASS})-[:{SAME_AS}]-(d2:{DRUG_CLASS})
            match (d1)-[:{INT_FOUND}]->(i:{INTERACTION_CLASS})<-[:{INT_FOUND}]-(d3:{DRUG_CLASS})
            match (i)-[:{RDF_TYPE}]->(type:{OWL_CLASS})
            where d1.key < d2.key and not (d2)-[:{INT_FOUND}]->()<-[:{INT_FOUND}]-(d3)
            with distinct d1.key as d1, d2.key as d2, d3.key as d3, type.key as type
            return type as {INTERACTION_CLASS}, count(d1) as count
        '''
        result = self.run_query(query)
        print('New pairs from owl:sameAs')
        print(result)


    def remove_isolated_nodes (self, pairs):
        query = f'''
            unwind $pairs as pair
            match (p:{INTERACTION_CLASS} {{key: pair}})
            detach delete p
        '''
        self.session.run(query=query, pairs=list(pairs))

        queries = [f'''
            match (s:{SENT_CLASS})
            where not (s)-[:{SENT_CON_INTER}]-()
            detach delete s
        ''', f'''
            match (d:{DRUG_CLASS}) 
            where not (d)-[:{ENT_FOUND_AS}]->(:{TOKEN_CLASS})<-[:{SENT_CON_TOKEN}]-(:{SENT_CLASS})
            detach delete d
        ''', f'''
            match (t:{TOKEN_CLASS})
            where not (t)<-[:{ENT_FOUND_AS}]-(:{DRUG_CLASS})
            detach delete t
        ''', f'''
            match (dn:{DATA_NODE})
            where not (dn)-[]-()
            delete dn
        ''']
        for query in queries:
            self.session.run(query=query)






