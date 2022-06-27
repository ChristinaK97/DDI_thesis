# FIELDS

E1 = 'e1'
E2 = 'e2'
E1_TYPE = 'e1_type'
E2_TYPE = 'e2_type'
PAIR_TYPE = 'pair_type'
DOC_ID = 'document_id'
SENT_ID = 'sentence_id'
E1_ID = 'e1_id'
E2_ID = 'e2_id'
PAIR_ID = 'pair_id'
SENT_TEXT = 'sentence_text'
E1_CHAR = 'e1_charOffset'
E2_CHAR = 'e2_charOffset'
DDI = 'ddi'


def ENT_CHAR(entity):
    return E1_CHAR if entity == E1 else E2_CHAR


def ENT_ID(entity):
    return E1_ID if entity == E1 else E2_ID


def ENT_TYPE(entity):
    return E1_TYPE if entity == E1 else E2_TYPE

# ---------------------------------------------------------------------
# PROPERTIES

RDF_NS = '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .'

RDF_TYPE = 'rdf_type'
RDF_S = 'rdf_subject'
RDF_P = 'rdf_predicate'
RDF_O = 'rdf_object'
RDF_TRIPLE = [RDF_S, RDF_P, RDF_O]

DOMAIN = 'rdfs_domain'
RANGE = 'rdfs_range'
SUBCLASS = 'rdfs_subClassOf'
SAME_AS = 'owl_sameAs'

KEY = 'key'
ENT_NAME = 'name'
START = 'start'
END = 'end'
SENT_SOURCE = 'source'
ENT_FOUND_AS = 'found_as'
INT_FOUND = 'interaction_found'
SENT_CON_INTER = 'contains_interaction'
SENT_CON_TOKEN = 'contains_token'
SENT_DOC = 'document'
WITH_TOKEN = 'with_token'
TOKEN_IN_INT = 'in_interaction'

# CLASSES
OWL_CLASS = 'owl_Class'
DRUG_CLASS = 'Drug_Class'
INTERACTION_CLASS = 'Interaction'
DRUG_CLASSES = {'drug', 'drug_n', 'brand', 'group'}
INTERACTION_CLASSES = {'negative', 'advise', 'effect', 'mechanism', 'int'}
TOKEN_CLASS = 'Token'
SENT_CLASS = 'Sentence'
PAIR_BLANK = '_:p'
DATA_NODE = 'Data_Node'

# DOMAINS / RANGES

DOMAIN_RANGE = {
    ENT_FOUND_AS :      {DOMAIN: DRUG_CLASS,        RANGE: TOKEN_CLASS},
    INT_FOUND:          {DOMAIN: DRUG_CLASS,        RANGE: INTERACTION_CLASS},
    ENT_NAME:           {DOMAIN: DRUG_CLASS,        RANGE: DATA_NODE},
    SAME_AS:            {DOMAIN: DRUG_CLASS,        RANGE: DRUG_CLASS},
    SENT_CON_INTER:     {DOMAIN: SENT_CLASS,        RANGE: INTERACTION_CLASS},
    SENT_CON_TOKEN:     {DOMAIN: SENT_CLASS,        RANGE: TOKEN_CLASS},
    SENT_DOC:           {DOMAIN: SENT_CLASS,        RANGE: DATA_NODE},
    SENT_TEXT:          {DOMAIN: SENT_CLASS,        RANGE: DATA_NODE},
    START:              {DOMAIN: TOKEN_CLASS,       RANGE: DATA_NODE},
    END:                {DOMAIN: TOKEN_CLASS,       RANGE: DATA_NODE},
    TOKEN_IN_INT:       {DOMAIN: TOKEN_CLASS,       RANGE: INTERACTION_CLASS},
    WITH_TOKEN:         {DOMAIN: INTERACTION_CLASS, RANGE: TOKEN_CLASS},
    SENT_SOURCE:        {DOMAIN: INTERACTION_CLASS, RANGE: SENT_CLASS}
}

INVERSE = {
    SENT_SOURCE:    SENT_CON_INTER,
    SENT_CON_INTER: SENT_SOURCE,

    WITH_TOKEN:     TOKEN_IN_INT,
    TOKEN_IN_INT:   WITH_TOKEN
}


def get_domain_range(predicate):
    try:
        dr = DOMAIN_RANGE[predicate]
        return dr[DOMAIN], dr[RANGE]
    except KeyError:
        return None, None

def get_inverse(predicate):
    return INVERSE.get(predicate, None)


def get_classes():
    return list(INTERACTION_CLASSES) + list(DRUG_CLASSES) \
        + [SENT_CLASS, TOKEN_CLASS, INTERACTION_CLASS, DRUG_CLASS]


def get_superClass(class_):
    if class_ in DRUG_CLASSES:
        return DRUG_CLASS
    elif class_ in INTERACTION_CLASSES:
        return INTERACTION_CLASS
    else:
        return OWL_CLASS

#


# OTHER
ANNOTATED = 'annotated'
LABEL = 'label'
e = 10e-6
