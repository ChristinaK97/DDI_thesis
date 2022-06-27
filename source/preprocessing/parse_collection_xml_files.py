import os
import xml.etree.ElementTree as et
from other.CONSTANTS import *


def parse_collection(directory):
    """
    Επεξεργασία όλων των documents μιας συλλογής
    :param directory: Το path του φακέλου της συλλογής
    :return:
        pairs ->    List από Dicts
                    όλων των pairs της συλλογής
    """
    documents = os.listdir(directory)
    pairs = list()
    for doc_path in documents:
        pairs = pairs + parse_xml(directory + "/" + doc_path)

    return pairs


def parse_xml(doc_path):
    """
    Επεξεργασία του αρχείου xml ενός document
    :param doc_path: Το path του xml
    :return:
        pairs ->    List από Dicts
                    των pairs του συγκεκριμένου doc
    """
    try:
        doc = et.parse(doc_path).getroot()
    except et.ParseError:  # άδειο έγγραφο
        return list()
    pairs = list()

    for sentence in doc:
        sentence_entities = {}  # Dict {entity.id : entity name}

        for entity in sentence.iter('entity'):
            entity = entity.attrib
            entity['text'] = parse_text(entity)
            sentence_entities[entity['id']] = entity

        for pair in sentence.iter('pair'):
            new_pair = parse_pair(pair.attrib, sentence_entities, doc.attrib['id'], sentence.attrib)
            if new_pair:
                pairs.append(new_pair)

    return pairs


def parse_pair(pair, sentence_entities, doc_id, sentence_attr):
    """
    Επεξεργασία ενός pair από ένα sentence
    :param doc_id: To id του εγγράφου
    :param sentence_attr: Dict με τα attributes του sequence από όπου προέρχεται το pair
    :param pair: Dict με τα attrib ενός pair
    :param sentence_entities: Dict {entity.id : entity name} μιας sentence
    :return:
        pair -> Dict του νέου pair
                ή αν e1 και e2 είναι η ίδια entity τότε None
    """
    # Πάρε τα ονόματα των entities από τα ids τους
    e1 = sentence_entities[pair['e1']]
    e2 = sentence_entities[pair['e2']]

    # Όχι ακμές-βρόχοι
    if e1['text'] == e2['text']:
        return None

    pair[DOC_ID] = doc_id
    pair[SENT_ID] = sentence_attr['id']
    pair[PAIR_ID] = pair.pop('id')
    pair[E1_ID] = e1['id']
    pair[E2_ID] = e2['id']

    pair[E1] = e1['text']
    pair[E2] = e2['text']
    pair[E1_TYPE] = e1['type']
    pair[E2_TYPE] = e2['type']

    # Αρνητική σχέση
    pair[PAIR_TYPE] = 'negative' if pair['ddi'] == 'false' else pair.pop('type')

    pair[SENT_TEXT] = sentence_attr['text']
    pair[E1_CHAR] = e1['charOffset']
    pair[E2_CHAR] = e2['charOffset']

    return pair


def parse_text(entity):
    """
    :param entity: Dict με τα attrib ενός entity μίας sentence
    :return: To attrib text της entity, επεξεργασμένο
    """
    return parse_name(entity['text'])


def parse_name(name) :
    """
    Μετατρέπει το όνομα σε lowercase
    και βγάζει τους κενούς χαρακτήρες στην αρχή και τέλος (trim)
    :param name: το όνομα μίας οντότητας
    :return: το όνομα επεξεργασμένο
    """
    return name.lower().strip()
