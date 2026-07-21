from scipy.stats import fisher_exact

# Level to binarize our output.
PREDICTION_THRESHOLD = 0.5

def print_statistics(tp: int, fp: int, fn: int, tn: int):
    # Step 5: Statistical Analysis
    # Using the counts obtained from Step 4, perform Fisher's exact test to determine the p-value.
    total = tp + fp + tn + fn
    predicted_positive = tp + fp
    overall_positive_rate = float(fn + tp) / total if total else 0.0
    precision = float(tp) / predicted_positive if predicted_positive else 0.0
    accuracy = float(tp + tn) / total if total else 0.0
    p_value = calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)

    # Step 6: Output
    # Print the following information:
    print(f'TN: {tn}')
    print(f'TP: {tp}')
    print(f'FN: {fn}')
    print(f'FP: {fp}')
    print(f'Overall positive rate: {overall_positive_rate}')
    print(f'Precision: {precision}')
    print(f'Accuracy: {accuracy}')
    print(f'P-value of precision: {p_value}')


def calculate_precision_p_value(tp: int, fp: int, fn: int, tn: int):
    """Return a one-sided Fisher p-value for positive prediction enrichment."""
    total = tp + fp + fn + tn
    predicted_positive = tp + fp
    if not total or not predicted_positive:
        return 1.0

    precision = tp / predicted_positive
    overall_positive_rate = (tp + fn) / total
    if precision <= overall_positive_rate:
        return 1.0

    # Rows are predicted class and columns are actual class.
    contingency_table = [[tp, fp], [fn, tn]]
    _, p_value = fisher_exact(contingency_table, alternative='greater')
    return p_value
