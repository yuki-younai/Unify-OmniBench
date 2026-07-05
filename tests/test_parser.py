from unify_omnibench.eval.parser import choices_to_index2ans, extract_choice_letter


def test_single_letter():
    assert extract_choice_letter("A") == "A"
    assert extract_choice_letter(" b ") == "B"


def test_boxed():
    assert extract_choice_letter("Reasoning... so \\boxed{C}.") == "C"
    assert extract_choice_letter("\\boxed{\\text{D}}") == "D"


def test_json_answer():
    assert extract_choice_letter('Some preamble {"answer":"B"}') == "B"


def test_standalone_letter():
    assert extract_choice_letter("I think the answer is C because ...") == "C"


def test_paren_letter():
    assert extract_choice_letter("Choice: (D) blue.") == "D"


def test_reverse_lookup():
    idx = choices_to_index2ans(["A. apple", "B. banana", "C. cherry", "D. date"])
    assert extract_choice_letter("definitely a banana", index2ans=idx) == "B"


def test_no_match():
    assert extract_choice_letter("") is None
    assert extract_choice_letter(None) is None  # type: ignore
    assert extract_choice_letter("0123") is None
