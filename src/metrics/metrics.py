import collections

def span_f1(predictions, references):
    """Computes Span based F1 score.

    Args:
      targets: list of strings or list of list of strings if multiple references
        are present.
      predictions: list of strings

    Returns:
      span f1 across all targets and predictions (Based on CoNLL script)
    """
    true_positives = collections.defaultdict(int)
    false_positives = collections.defaultdict(int)
    false_negatives = collections.defaultdict(int)

    def tags_to_spans(tag_sequence, delimiter='$$'):
        """Extract spans from IOB1 or BIO tags."""
        tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
        tags_entities = []
        for tag_entity in tag_sequence_split:
            tag_entity_split = tag_entity.split(':')
            if len(tag_entity_split) != 2:
                continue
            tag = tag_entity_split[0].strip()
            entity = tag_entity_split[1].strip()
            tags_entities.append((tag, entity))
        return tags_entities

    def compute_f1_metrics(true_positives, false_positives, false_negatives):
        precision = float(true_positives) / float(true_positives + false_positives +
                                                  1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives +
                                               1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    for target, pred in zip(references, predictions):
        gold_spans = tags_to_spans(target)
        predicted_spans = tags_to_spans(pred)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                false_positives[span[0]] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            false_negatives[span[0]] += 1

    _, _, f1_measure = compute_f1_metrics(
        sum(true_positives.values()), sum(false_positives.values()),
        sum(false_negatives.values()))

    return {'span_f1': f1_measure * 100}