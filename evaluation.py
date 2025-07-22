# evaluation.py
from rag_pipeline import evaluate_groundedness

test_cases = [
    {"query": "বাংলাদেশের জাতীয় সংগীত কে রচনা করেছেন?", "expected": "রবীন্দ্রনাথ ঠাকুর"},
    {"query": "বাংলাদেশের রাজধানী কোথায়?", "expected": "ঢাকা"},
]

for test in test_cases:
    result = evaluate_groundedness(test["query"], test["expected"])
    print(result)
