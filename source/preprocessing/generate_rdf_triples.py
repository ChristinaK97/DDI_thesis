import re
import pandas as pd
from io import StringIO

from other.CONSTANTS import *
from other.file_paths import processed_data_path
from source.preprocessing.handle_multitype_entities import get_blank_node_id as blank_node


def make_rdf_triples(train, ddi, multitype_entities, *other_triples):
    """
    Καλεί τις υπόλοιπες μεθόδους.
    Δημιουργεί το αρχείο csv με τις RDF τριπλέτες από τη συλλογή.
    Ακόμα, το αρχείο ttl με τις τριπλέτες εκφρασμένες σε σύνταξη Turtle.
    :param ddi: Το πλήρες dataframe με τα αρχικά δεδομένα
    :param multitype_entities: Ένα σετ με τα ονόματα των οντοτήτων (entity name/text)
           οι οποίες εμφανίστηκαν με πολλαπλούς τύπους (και δεν οφείλονταν σε annotation error)
    :return: Dataframe με τις RDF τριπλέτες. Πεδία <RDF_S, RDF_P, RDF_O>
    """
    folder = processed_data_path(train)
    try:
        triples = pd.read_csv(folder + 'triples.csv')

    except FileNotFoundError:  # αν δεν έχει φτιαχτεί το αρχείο, το φτιάχνει από το ddi

        triples = parse_ddi(ddi, multitype_entities)
        triples.drop_duplicates(inplace=True, ignore_index=True)

        if len(other_triples) > 0 :
            triples = pd.concat(list(other_triples) + [triples])

        triples.sort_values(RDF_TRIPLE, inplace=True, ignore_index=True)
        triples.to_csv(folder + 'triples.csv', sep=',', encoding='utf-8', index=False)

        triples_to_turtle_parser(triples, folder)
    return triples


def parse_ddi(ddi, multitype_entities):
    """
    Καλείται μέσω της make_rdf_triples για να δημιουργήσει το DF με όλες τις
    τριπλέτες που δίνει το DF ddi των αρχικών δεδομένων.
    :param ddi: Όρισμα της make_rdf_triples
    :param multitype_entities: Όρισμα της make_rdf_triples
    :return: DF των τριπλετών. Περιέχει διπλότυπα.
    """
    t = []
    #unique_pairs = {}

    for index, line in ddi.iterrows():

        pair = line[PAIR_ID]

        """
        # Τα ονόματα και οι τύποι των δύο οντοτήτων, και ο τύπος της σχέσης που τις συνδέει
        # Στην ουσία μια συγκεκριμένη σχέση μεταξύ δύο συγκεκριμένων οντοτήτων
        line_str = f"{line[E1]}{line[E2]}{line[E1_TYPE]}{line[E2_TYPE]}{line[PAIR_TYPE]}"
        # Αν δεν έχεις ξαναδεί τη συγκεκριμένη συσχέτιση
        if line_str not in unique_pairs:
            # Νέα συσχέτιση που αναπαριστάται από κόμβο _:p{Α/Α}
            pair = PAIR_BLANK + str(index)
            unique_pairs[line_str] = pair
        else:
            # Ανάκτησε τον κόμβο της συγκεκριμένης συσχέτισης
            pair = unique_pairs[line_str]
        """

        # Το e1.text : όνομα της οντότητας και
        # τριπλέτα <_:{e1.text}_{e1.type}, name, {e1.text}> αν είναι multitype
        sub_name, bl_node_name = parse_entity(E1, line, multitype_entities)
        if bl_node_name is not None: t.append(bl_node_name)

        obj_name, bl_node_name = parse_entity(E2, line, multitype_entities)
        if bl_node_name is not None: t.append(bl_node_name)

        t += [
            # e1 (subject) triples
            (sub_name, RDF_TYPE, line[E1_TYPE]),
            (sub_name, ENT_FOUND_AS, line[E1_ID]),
            (sub_name, INT_FOUND, pair),

            # e2 (object) triples
            (obj_name, RDF_TYPE, line[E2_TYPE]),
            (obj_name, ENT_FOUND_AS, line[E2_ID]),
            (obj_name, INT_FOUND, pair),

            # interaction blank node triples
            (pair, RDF_TYPE, line[PAIR_TYPE]),
            (pair, SENT_SOURCE, line[SENT_ID]),

            # sentence triples
            (line[SENT_ID], RDF_TYPE, SENT_CLASS),
            (line[SENT_ID], SENT_CON_INTER, pair),
            (line[SENT_ID], SENT_DOC, '\"' + line[DOC_ID] + '\"'),
            (line[SENT_ID], SENT_TEXT, '\"' + line[SENT_TEXT] + '\"')
        ]
        # <sentence.id, contains_token, token.id>
        # : H πρόταση περιέχει τα tokens που αντιστοιχούν στις δύο οντότητες
        # <pair.id, with_token, token.id}
        for token in [line[E1_ID], line[E2_ID]]:
            t.append((line[SENT_ID], SENT_CON_TOKEN, token))
            t.append((pair, WITH_TOKEN, token))

        # token triples
        for E in [E1, E2]:
            t += make_token_entry(line, E)

    t = pd.DataFrame(t, columns=RDF_TRIPLE)
    return t


def parse_entity(entity, line, multitype_entities):
    """
    Υπεύθυνη για την επεξεργασία του ονόματος της οντότητας (e.text)
    :param entity: Πεδίο E1 ή E2 του line/ddi όπου είναι το όνομα της οντότητας
                   (αναλόγως θα προσπελάσει το line)
    :param line: Μια εγγραφή του DF ddi των αρχικών δεδομένων.
                 Το εκάστοτε sentence.pair που επεξεργάζεται.
    :param multitype_entities: Ένα σετ με τα ονόματα των οντοτήτων (entity name/text)
           οι οποίες εμφανίστηκαν με πολλαπλούς τύπους (και δεν οφείλονταν σε annotation error)
    :return: Αν η οντότητα δεν είναι multitype:
                - το όνομά της όπως είναι στο line και ένα None
             Αλλιώς:
                - _:{e1.text}_{e1.type} για όνομα του κόμβου της multitype οντότητας
                - τριπλέτα <_:{e1.text}_{e1.type}, name, {e1.text}> που δείχνει το όνομα
                  της οντότητας που αντιστοιχεί σε αυτόν τον blank node
    """
    entity_name = line[entity]
    triple = None

    if entity_name in multitype_entities:
        blank_name = blank_node(entity_name, line[ENT_TYPE(entity)])
        triple = (blank_name, ENT_NAME, entity_name)
        entity_name = blank_name
    return entity_name, triple


def make_token_entry(line, entity):
    """
    Επεξεργάζεται τα πεδία start και end που αντιστοιχούν σε ένα token
    μιας οντότητας.
    :param line: Μια εγγραφή του DF ddi των αρχικών δεδομένων.
    :param entity: Πεδίο E1 ή E2 του line/ddi όπου είναι το όνομα της οντότητας
                   (αναλόγως θα προσπελάσει το line)
    :return: Μια λίστα με τριπλέτες <{e.id}, start/end, {start/end}>
             Τα όρια του token μέσα στο sentence
    """
    indexes_list = []
    token_key = line[ENT_ID(entity)]
    indexes_list.append((token_key, RDF_TYPE, TOKEN_CLASS))

    # Πιθανές μορφές πεδίου: {start}-{end} ή {start1}-{end1};{start2}-{end2}
    indexes = re.split(';|-', line[ENT_CHAR(entity)])

    # Για κάθε ζεύγος start και end φτιάξε τριπλέτες:
    # <{e.id}, start, {start}>, <{e.id}, end, {end}>
    for i in range(0, int(len(indexes) / 2)):
        indexes_list.append((token_key, START, indexes[2 * i]))
        indexes_list.append((token_key, END, indexes[2 * i + 1]))

    return indexes_list

# -------------------------------------------------------------------------------------


def triples_to_turtle_parser(t, folder):
    """
    Δημιουργεί το αρχείο turtle.ttl με τις τριπλέτες εκφρασμένες σε σύνταξη Turtle.
    :param t: Το σύνολο των τριπλέτων του γράφου, χωρίς διπλότυπα και
              ταξινομημένο ως προς RDF_S, RDF_P (τουλάχιστον)
    :return: -
    """
    data = StringIO()
    data.write(RDF_NS + '\n\n')

    s_not_written, p_not_written = True, True

    # Για κάθε τριπλέτα
    for i in t.index:

        if s_not_written:
            # Γράψε το subject
            data.write(t[RDF_S][i] + '\t')
            s_not_written = False
        else:
            data.write('\t')

        if p_not_written:
            # Γράψε το predicate
            data.write(t[RDF_P][i] + '\t')
            p_not_written = False
        else:
            data.write('\t')

        try:
            # Γράψε κάθε object που συμμετέχει σε τριπλέτα με
            # τα συγκεκριμένα subject και predicate
            data.write(t[RDF_O][i])
        except TypeError:
            print(t[RDF_O][i])

        # 1. Η επόμενη τριπλέτα έχει διαφορετικό subject?
        # 2. Αν η επόμενη τριπλέτα έχει διαφορετικό predicate :
        #       3. Θα πρέπει να γράψεις το predicate i+1 στο επόμενο loop
        #       4. Αν δεν άλλαζει μετά το subject :
        #           5. προχώρα να δηλώσεις το επόμενο predicate για αυτό (διαχωριστικό ;)
        # 6. Αν η επόμενη έχει ίδιο predicate :
        #       7. πρόχώρα να γράψεις το επόμενο object/τιμή της ιδιότητας (διαχωριστικό ,)
        # 8. Αν έφτασες στο τέλος του DF, πρέπει να γράψεις το τελευταίο διαχωριστικό .
        # 9. Αν η επόμενη τριπλέτα έχει διαφορετικό subject :
        #       10. Θα πρέπει να γράψεις το subject i+1 στο επόμενο loop
        #       11. και το διαχωριστικό . -> δηλώνει την τελευταία τριπλέτα του subject
        try:
            s_change = t[RDF_S][i] != t[RDF_S][i + 1]   # 1

            if t[RDF_P][i] != t[RDF_P][i + 1]:          # 2
                p_not_written = True                    # 3
                if not s_change:                        # 4
                    data.write(" ;\n")                  # 5
            else:                                       # 6
                data.write(" ,\n\t\t")                  # 7
        except KeyError:                                # 8
            s_change = True
        if s_change:                                    # 9
            s_not_written = True                        # 10
            data.write(" .\n\n")                        # 11

    open(folder + "turtle.ttl", "w").write(data.getvalue())
