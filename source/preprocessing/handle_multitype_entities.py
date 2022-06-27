import pandas as pd

from other.CONSTANTS import RDF_S, RDF_O, SAME_AS, DOC_ID, E1, E1_TYPE, E2, E2_TYPE, RDF_TRIPLE, RDF_TYPE, RDF_P


def get_blank_node_id(E, T):
    """
    Το key του blank node της multitype οντότητας
    :param E: Το όνομα της οντότητας όπως φαίνεται στο πεδίο E του line
    :param T: Ο τύπος με τον οποίο φαίνεται στη συγκεκριμένη συσχέτιση
    :return: _:{e1.text}_{e1.type}
    """
    return f"_:{E}_{T}"


def find_multitype_entities(ddi):
    # > error_thrs % εμφανίσεις για μία κλάση -> θεωρείται σφάλμα
    # => αντικατάσταση με την επικρατούσα κλάση
    error_thrs = 5
    multitype_entities = {}

    # [E,TYPE,counts] : Πλήθος εμφανίσεων κάθε οντότητας για κάθε τύπο της
    ent_types = pd.concat([ddi[[DOC_ID, E1, E1_TYPE]].rename(columns={E1: 'E', E1_TYPE: 'TYPE'}),
                           ddi[[DOC_ID, E2, E2_TYPE]].rename(columns={E2: 'E', E2_TYPE: 'TYPE'})
                           ]).groupby(['E', 'TYPE']).size().reset_index(name='counts')

    # [E,counts] : πλήθος τύπων που λαμβάνει κάθε οντότητα
    ent_count_types = ent_types[['E', 'TYPE']].groupby(['E']).size().reset_index(name='counts')

    # 1-class entities : 2052, 2-class : 59, 3-class: 1
    # print(ent_count_types.groupby(['counts']).count())

    # [index,E] : οι οντότητες με περισσότερους από έναν τύπους
    ent_count_types = ent_count_types.loc[ent_count_types['counts'] > 1]['E'] \
        .reset_index(drop=True)

    for entity in ent_count_types:
        type_instances = ent_types.loc[ent_types['E'] == entity][['TYPE', 'counts']].reset_index(drop=True)
        # πλήθος εμφανίσεων κάθε οντότητας ανεξάρτητα του τύπου
        ent_instances = type_instances['counts'].sum()

        type_instances['perc']  = type_instances['counts'] / ent_instances * 100
        type_instances['error'] = type_instances['perc'] <= error_thrs

        # αν υπάρχει σφάλμα
        if True in type_instances['error'].values:
            # ορθός τύπος αυτός που έχει το μεγαλύτερο ποσοστό εμφανίσεων
            correct_type = type_instances.iloc[type_instances['perc'].idxmax()]['TYPE']

            # αλλαγή των εσφαλμένων τύπων με τον ορθό
            for error_type in type_instances.loc[type_instances['error']]['TYPE'].values:
                for E, T in [(E1, E1_TYPE), (E2, E2_TYPE)]:
                    try:
                        ddi.loc[(ddi[E] == entity) & (ddi[T] == error_type), T] = correct_type
                    # (E,T) δεν είχε εγγραφές (entity, error_type)
                    except ValueError:
                        pass

        # αλλιώς η οντότητα είναι όντως πολλαπλών τύπων
        else:
            multitype_entities[entity] = set(type_instances['TYPE'])

    return multitype_entities


def multitype_sameAs(sameAs, ddi, multitype_entities):

    def get_types(e, e_mult):
        return multitype_entities.get(e) if e_mult else singletype(e)
    def get_blank(e, e_mult):
        return blank_node_names(e, common_types) if e_mult else [e]
    def singletype(e):
        e_types = ddi[ddi[E1] == e][E1_TYPE].values
        if len(e_types) == 0:
            e_types = ddi[ddi[E2] == e][E2_TYPE].values
        return set(e_types)
    def blank_node_names(e, types):
        return [get_blank_node_id(e, t) for t in types]
    def make_triple(e1_name, e2_name):
        return e1_name, SAME_AS, e2_name

    drop_index = []
    new_triples = []

    for i, pair in sameAs.iterrows():

        e1, e2 = pair[RDF_S], pair[RDF_O]

        e1_mult, e2_mult = (e in multitype_entities for e in [e1, e2])

        if not (e1_mult or e2_mult):
            continue
        e1_types = get_types(e1, e1_mult)
        e2_types = get_types(e2, e2_mult)

        common_types = e1_types.intersection(e2_types)

        if len(common_types) > 0 :
            e1_blanks = get_blank(e1, e1_mult)
            e2_blanks = get_blank(e2, e2_mult)

            [new_triples.append(make_triple(e1_bl, e2_bl))
                for e1_bl, e2_bl in zip(e1_blanks, e2_blanks)]

        drop_index.append(i)

    # [print(i) for i in new_triples]
    sameAs.drop(sameAs.index[drop_index], inplace=True)
    new_triples = pd.DataFrame(new_triples, columns=RDF_TRIPLE)
    sameAs = pd.concat([sameAs, new_triples]).reset_index(drop=True)

    return sameAs




