import re

import numpy as np
import pandas as pd
from other.CONSTANTS import *
from source.database_pcg.n4j_pcg.query_neo4j import Query_Neo4j as Neo4j


class NegativeFiltering:

    def __init__(self, train):

        self.n4j = Neo4j(train=train)

        sentences_init = self.n4j.run_query(self.n4j.q_collect_sentences())

        self.sentences = pd.Series([s[0][1:-1] for s in sentences_init])
        self.named_entities = [sorted(s[1], key=lambda i: i[START]) for s in sentences_init]

        self.masked_sentences = self.mask_sentences()

        self.remove_filtered_pairs()
        self.n4j.close()



    def remove_filtered_pairs(self):
        """
        Τρέχει όλες τις άλλες μεθόδους, ώστε να βρεθούν τα τετριμμένα ζεύγη
        και να αφαιρεθούν από τη βάση
        filtered_pairs:
            set από str. Οι κόμβοι των τετριμμένων ζευγών.
                 Αναφέρονται ως '_:p{id}'
                 Έξοδος της q_interaction_nodes.
        """
        filtered_pairs = \
            self.q_interaction_nodes(
                self._find_filtered_entity_pairs(
                    self.match_sentences_with_regex(
                        self.make_regex_list()
                    )))
        self.n4j.remove_isolated_nodes(filtered_pairs)



    def mask_sentences(self):
        """
        Στα sentences του dataset, αντικαθιστά τα tokens που αντιστοιχούν σε
        σημειωμένες οντότητες (named entities), με το χαρακτήρα 'φ',
        ώστε να εντοπιστούν μέσω regex.
        :return: List με len = # προτάσεων στο dataset
                 Η κάθε πρόταση, αφού έχει γίνει αυτή η αντικατάσταση
        """
        masked_sentences = []
        for s, s_ne in zip(self.sentences, self.named_entities):
            for t in s_ne:
                t_len = t[END] - t[START] + 1
                s = ('φ' * t_len).join([s[:t[START]], s[t[END] + 1:]])
            masked_sentences.append(s)
        return masked_sentences


    def make_regex_list(self):
        """
        Ορίζονται οι regex που θα εφαρμοστούν στις προτάσεις, ώστε να εντοπιστούν
        μοτίβα που φανερώνουν τετριμμένες αρνητικές (negative) σχέσεις/ζεύγη.
        :return: List με αντικείμενα pattern αυτών των regex.
        """
        regex = []
        # Οι οντότητες αναφέρονται με το regex φ+

        # Αναγραφή/Λίστα από τουλάχιστον 3 οντότητες. Εξαιρούνται σημεία στίξης () []
        # που θα εντοπιστούν από παρακάτω κανόνα
        """
        ' DRUG2, DRUG3, DRUG4'>
        ' DRUG26;DRUG27,DRUG28 , DRUG29'>
        """
        regex.append(
            re.compile(r'\s*φ+'
                       r'(\s*[^\w\s()\[\]]\s*φ+){2,}')
        )
        # Λίστα από οντότητες διαχωρισμένες με σημεία στίξης, και επιπλέον στο τέλος της λίστας
        # σύνδεσμος and ή or και ακολουθεί μία τελευταία οντότητα. Συνολικά απαιτείται η λίστα
        # να αποτελείται από 3 τουλάχιστον οντότητες. Εξαιρούνται σημεία στίξης () []
        # που θα εντοπιστούν από παρακάτω κανόνα
        """
        ' DRUG2, DRUG3, DRUG4, and DRUG5'>
        ' DRUG18, DRUG19, DRUG20 and DRUG21'>
        ' DRUG22, DRUG23 and DRUG25'>
        ' DRUG26;DRUG27,DRUG28 , DRUG29 ,and DRUG30'>
        """
        regex.append(
            re.compile(r'\s*φ+'
                       r'(\s*[^\w\s()\[\]]\s*φ+)+'

                       r'(\s*[^\w\s()]?\s*'
                       r'(and|or)\s*φ+)')
        )
        # Οντότητα που ακολουθείται από λίστα συνωνύμων, υπο/υπερ-κατηγοριών
        # που αναγράφεται προαιρετικά μέσα σε παρένθεση, αλλά δηλώνεται μέσω ενός
        # από τα eg/ie/such as/a/an. Η λίστα μπορεί να αποτελείται από >=1 οντότητες
        # και προαιρετικά να περιλαμβάνει και and ή or στην τελευταία.
        """
        ' DRUG1 (e.g., DRUG2, DRUG3, DRUG4, and DRUG5)'>
        ' DRUG6 (eg DRUG7, DRUG8, DRUG9'>
        ' DRUG50 (eg DRUG51 and DRUG52'>
        ' DRUG47 (such as DRUG48 and DRUG49)'>
        ' DRUG12 ( e.g., DRUG13)'>
        ' DRUG16  e.g., DRUG17'>
        """
        regex.append(
            re.compile(r'\s*φ+\s*'
                       r'[([]?\s*[^\w\s]*\s*'
                       r'(e.g|eg|e. g|i.e|ie|such as|a|an)'

                       r'(\s*[^\w\s]*(and|or)*\s*φ+)+'

                       r'(\s*[^\w\s]?\s*'
                       r'(and|or)\s*φ+)?'
                       r'[^\w]*')
        )
        # Οντότητα που ακολουθείται από λίστα συνωνύμων, υπο/υπερ-κατηγοριών
        # που αναγράφεται υποχρεωτικά μέσα σε παρένθεση.
        # Η λίστα μπορεί να αποτελείται από >=1 οντότητες
        # και προαιρετικά να περιλαμβάνει και and ή or στην τελευταία.
        """
        ' DRUG37 (DRUG38, DRUG39, DRUG40)'>
        ' DRUG37 (DRUG41, DRUG42 ,and DRUG43)'>
        ' DRUG44 (DRUG45 and DRUG46)'>
        ' DRUG10 (DRUG11)'>
        ' DRUG14 [DRUG15)'>
        """
        regex.append(
            re.compile(r'\s*φ+\s*'
                       r'[([]'
                       r'\s*φ+'

                       r'('
                       r'(\s*[^\w\s]\s*φ+)*'

                       r'(\s*[^\w\s]?\s*'
                       r'(and|or)\s*φ+)?'
                       r')*'

                       r'[])]?')
        )

        return regex


    def match_sentences_with_regex(self, regex_list):
        """
        Εκτελεί το ταίριασμα των sentences με τα regex.
        1. Για κάθε sentences i και για κάθε regex pattern :
           2. Ελέγχει αν υπάρχει ταίριασμα και κρατάει τα όρια (start, end) σε λίστα

        :param regex_list: List με τα αντικείμενα pattern των regex. Έξοδος της make_regex_list
        :return: Dict  key = i το index της sentence,
                       value = List[Tuple(start, end)] με τα όρια των matcher της πρότασης
        """
        matches = {}
        for i, ms in enumerate(self.masked_sentences):  # 1
            sentence_matches = []
            for pattern in regex_list:
                pattern_matches = pattern.finditer(ms)  # 2
                [sentence_matches.append(match.span()) for match in pattern_matches]
            if len(sentence_matches) > 0:
                matches[i] = self._merge_spans(sentence_matches)

        return matches


    def _merge_spans(self, sentence_matches):
        """
        Συγχωνεύει διαστήματα που παρουσιάζουν επικάλυψη ή είναι διαδοχικά για μία πρόταση
        :param sentence_matches: List[Tuple(start, end)] με τα όρια των matcher της πρότασης
        :return: List[Tuple(start, end)] συγχωνευμένα
        """
        """
        1. Ταξινομεί τη λίστα των τριπλετών πρώτα ως προς start, και αν start είναι ίσα, 
           τότε και ως προς end
        2. Τα όρια που πρώτου match μέσα στην πρόταση
        3. Για κάθε match μετά το πρώτο
        4. Τα όρια του next match
        5. Αν το start του next είναι μικρότερο από end του τρέχοντος πχ c=(1,5) n=(3,_)
           ή τα διαστήματα είναι διαδοχικά πχ c=(1,5) n=(6,_) :
                6. Κρατάει το end του πιο ευρύ διαστήματος πχ (1,5) (3,7) => (1,7)
                                                              (1,5) (3,4) => (1,5)
        7. Αλλιώς αν τα διαστήματα δεν έχουν επικάλυψη :
           8. Κρατάει το διάστημα με τα όρια που βρήκε
           9. Προχωρά στο επόμενο διάστημα
        10. Κρατάει το διάστημα με τα όρια που βρήκε σε περίπτωση που δεν άλλαξε στο επόμενο (7) 
        """
        spans = sorted(sentence_matches, key=lambda x: (x[0], x[1],))  # 1

        pairs = []
        Cstart, Cend = spans[0]          # 2
        for i in range(len(spans) - 1):  # 3
            Nstart, Nend = spans[i + 1]  # 4

            if Nstart < Cend or Cend + 1 == Nstart:  # 5
                Cend = max(Cend, Nend)               # 6
            else:                                    # 7
                pairs.append((Cstart, Cend))         # 8
                Cstart, Cend = Nstart, Nend          # 9
        pairs.append((Cstart, Cend))                 # 10
        return pairs


    def _find_filtered_entity_pairs(self, matches):
        """
        Με βάση τα ταιριάσματα που προέκυψαν, εντοπίζει τα τετριμμένα ζεύγη οντοτήτων
        και τις προτάσεις από τις οποίες αυτά προέρχονται.
        :param matches: Dict  key = i το index της sentence,
               value = List[Tuple(start, end)] με τα όρια των matcher της πρότασης.
               Έξοδος της match_sentences_with_regex
        :return: List[Dict{E1, E2 : Token των οντοτήτων που αποτελούν ένα τετριμμένο
                 ζεύγος, SENT_CL : Το id της πρότασης από όπου προέκυψε το ζεύγος}]
        """
        """
        1. Για κάθε sentence i που ταίριαξε (έχει matches) με τουλ μία regex :
           2. Για κάθε τέτοιο ταίριασμα με όρια start και end μέσα στο sentence :
              3. Για κάθε σημειωμένη οντότητα (named entity) μέσα στην πρόταση :
                 4. Αν match.start <= ne.START και ne.END <= match.end, δηλ. αν
                    η οντότητα περιλαμβάνεται εντός των ορίων του ταιριάσματος :
                    5. Η οντότητα συμμετέχει σε τουλ ένα τετριμμένο ζεύγος
              6. Κάθε ζεύγος οντοτήτων που βρέθηκαν μέσα στο ίδιο match
                 (ταίριαξαν με κάποια regex), θεωρείται τετριμμένο.
        7. Διπλότυπα ζεύγη απορρίπτονται καθώς και θεωρείται (x,y) == (y,x)  
        """
        pairs = []
        for i, sentence_matches in matches.items():     # 1

            for match in sentence_matches:              # 2
                start, end = match
                match_entities = []

                for entity in self.named_entities[i]:   # 3
                    if start <= entity[START] and entity[END] <= end:   # 4
                        match_entities.append(entity[TOKEN_CLASS])     # 5

                for k in range(len(match_entities)):                    # 6
                    for l in range(k + 1, len(match_entities)):
                        pairs.append((match_entities[k], match_entities[l],
                                      match_entities[l][:match_entities[l].rfind('.')]))

        pairs = pd.DataFrame(pairs, columns=[E1, E2, SENT_CLASS])
        pairs = pairs.loc[
            pd.DataFrame(np.sort(pairs[[E1, E2]], 1), index=pairs.index)    # 7
                .drop_duplicates(keep='first').index
        ]
        pairs = pairs.to_dict('records')
        return pairs


    def q_interaction_nodes(self, pairs):
        """
        Βρίσκει τα τετριμμένα ζεύγη που αναφέρονται στη βάση μέσω των κενών κόμβων _:p{id}
        :param pairs: List[Dict{E1, E2 : Token των οντοτήτων που αποτελούν ένα τετριμμένο
               ζεύγος, SENT_CL : Το id της πρότασης από όπου προέκυψε το ζεύγος}]
               Έξοδος της _find_filtered_entity_pairs
        :return: set από str. Οι κόμβοι των τετριμμένων ζευγών.
                 Αναφέρονται ως '_:p{id}'
        """
        """
        1. Βρίσκει τα ζεύγη των οντοτήτων d1 και d2, που συνδέονται με τετριμμένη σχέση i
           (κόμβος _:p{id}) τύπου i_type, η οποία προέκυψε από την πρόταση s. 
           Οι δύο οντότητες αναφέρονται με τα Tokens d1_token και d2_token
        2. Επίσης τετριμμένη σχέση θεωρείται κάποια που εμφανίζεται μεταξύ οντοτήτων
           που συνδέονται μέσω sameAs, με την λογική ότι κάποια ουσία δεν αλληλεπιδρά 
           με τον εαυτό της. Συμπληρωματικά κατά την προεπεξεργασία (parse των xml)
           είχαν αφαιρεθεί τα σχέσεις που αναφέρονται στην ίδια οντότητα (ίδιο όνομα)
        """
        print('Query Neo4j...')

        query = f'''
            unwind $pairs as pair
            match (d1:{DRUG_CLASS})-[:{ENT_FOUND_AS}]->(d1_token:{TOKEN_CLASS})
            match (d2:{DRUG_CLASS})-[:{ENT_FOUND_AS}]->(d2_token:{TOKEN_CLASS})
            match (s:{SENT_CLASS})<-[:{SENT_SOURCE}]-(i:{INTERACTION_CLASS})-[:{RDF_TYPE}]->(i_type:{OWL_CLASS})
            
            where d1_token.key = pair.{E1} and d2_token.key = pair.{E2} and s.key = pair.{SENT_CLASS}
                and (d1)-[:{INT_FOUND}]->(i)<-[:{INT_FOUND}]-(d2)
            return d1.key as {E1}, d2.key as {E2}, i.key as {INTERACTION_CLASS}, i_type.key as {PAIR_TYPE}
        '''
        filtered_pairs = self.n4j.session.run(query=query, pairs=pairs).values()
        self.print_stats(filtered_pairs)

        sameAs_filtered_pairs = self._sameAs_filtered_pairs()
        filtered_pairs += sameAs_filtered_pairs
        filtered_pairs = {p_node[2] for p_node in filtered_pairs}
        filtered_pairs = filtered_pairs.union(self._same_sentence_contradiction())

        print('Total filtered pairs = ', len(filtered_pairs))

        return filtered_pairs


    def _sameAs_filtered_pairs(self):
        query = f'''
            match (d1:{DRUG_CLASS})-[:{SAME_AS}]-(d2:{DRUG_CLASS})
            match (d1)-[:{INT_FOUND}]->(i:{INTERACTION_CLASS})<-[:{INT_FOUND}]-(d2)
            match (i)-[:{RDF_TYPE}]->(i_type:{OWL_CLASS})
            return d1.key as {E1}, d2.key as {E2}, i.key as {INTERACTION_CLASS}, i_type.key as {PAIR_TYPE}
        '''
        sameAs_filtered_pairs = self.n4j.run_query(query)
        self.print_stats(sameAs_filtered_pairs)
        return sameAs_filtered_pairs



    def _same_sentence_contradiction(self):

        query = f'''
            match (token1:{TOKEN_CLASS})<-[:{ENT_FOUND_AS}]-(d:{DRUG_CLASS})-[:{ENT_FOUND_AS}]->(token2:{TOKEN_CLASS})
            match (token1)<-[:{SENT_CON_TOKEN}]-(s:{SENT_CLASS})-[:{SENT_CON_TOKEN}]->(token2)
            match (token1)-[:{END}]->(e1:{DATA_NODE})
            match (token2)-[:{END}]->(e2:{DATA_NODE})
            match (s)-[:{SENT_TEXT}]->(text:{DATA_NODE})
            where token1.key < token2.key
            return text.key as {SENT_CLASS}, d.key as d, collect([{{{TOKEN_CLASS}:token1.key, {END}: toInteger(e1.key)}}, 
                                                                  {{{TOKEN_CLASS}:token2.key, {END}: toInteger(e2.key)}}]) as token_pairs
            order by {SENT_CLASS}
        '''
        sentences_init = self.n4j.run_query(query)
        sentences = []

        for record in sentences_init:
            sentence = record[0][1:-1]
            tokens = {}
            for r in record[2]:
                for token in r:
                    old_value = tokens.get(token[TOKEN_CLASS], -1)
                    if old_value == -1 or old_value < token[END]:
                        tokens[token[TOKEN_CLASS]] = token[END]

            sentences.append((sentence, tokens))

        rm_tokens = []
        for sentence, tokens in sentences:

            sep_index = sentence.find(':')
            if sep_index == -1:
                sep_index = sentence.find(' - ')
                if sep_index == -1: continue

            for token, end in tokens.items():
                if 0 < sep_index - end < 35:
                    rm_tokens.append(token)

        query = f'''
            unwind $rm_tokens as t_key
            match (type:{OWL_CLASS})<-[:{RDF_TYPE}]-(p:{INTERACTION_CLASS})-[:{WITH_TOKEN}]->(token:{TOKEN_CLASS})
            where token.key = t_key
            return t_key as {E1}, t_key as {E2}, p.key as {INTERACTION_CLASS}, type.key as {PAIR_TYPE}
        '''
        pairs = self.n4j.session.run(query=query, rm_tokens=rm_tokens).values()
        self.print_stats(pairs)

        return {pair[2] for pair in pairs}



    def print_stats(self, filtered_pairs):
        """
        Τυπώνει στοιχεία
        :param filtered_pairs: List[Dict{
                E1, E2 : Ονόματα των οντοτήτων που αποτελούν ένα τετριμμένο ζεύγος,
                INTERACTION : blank node _:p{id},
                PAIR_TYPE : Ο τύπος του τετριμμένου ζεύγους (ddi type)
            }]
        """
        df = pd.DataFrame(filtered_pairs, columns=[E1, E2, INTERACTION_CLASS, PAIR_TYPE])
        df.drop_duplicates(subset=[INTERACTION_CLASS], inplace=True, ignore_index=True)
        print('# pairs =', df.shape[0])
        print(df[[PAIR_TYPE, INTERACTION_CLASS]].groupby(PAIR_TYPE).count())

        non_negative = df[df[PAIR_TYPE] != 'negative'].to_dict('records')

        if len(non_negative) > 0 :
            query = f'''
                unwind $non_negative as nn
                match (p:{INTERACTION_CLASS})-[:{SENT_SOURCE}]->(s:{SENT_CLASS})-[:{SENT_TEXT}]->(text:{DATA_NODE})
                where p.key = nn.{INTERACTION_CLASS}
                return p.key, nn.{E1}, nn.{E2}, nn.{PAIR_TYPE}, text.key
            '''
            non_negative = self.n4j.session.run(query=query, non_negative=non_negative).values()
            [print(nn) for nn in non_negative]


