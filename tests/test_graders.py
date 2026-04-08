import pytest
from env.graders.email_grader import EmailGrader
from env.graders.code_grader import CodeGrader
from env.graders.data_grader import DataGrader

def test_email_grader():
    grader = EmailGrader()
    assert grader.grade("urgent", "urgent") == 1.0
    assert grader.grade("urgent", "spam") == 0.0
    assert grader.grade("invalid", "urgent") == 0.0

def test_code_grader_detect_bug():
    grader = CodeGrader()
    expected_bugs = ["null pointer exception", "infinite loop"]
    
    # Keyword 'infinite' should match 'infinite loop'
    assert grader.grade("detect_bug", {"description": "There is an infinite loop"}, expected_bugs, []) == 1.0
    # No match
    assert grader.grade("detect_bug", {"description": "Syntax error"}, expected_bugs, []) == 0.0

def test_code_grader_suggest_fix():
    grader = CodeGrader()
    expected_fixes = ["Add a break condition", "Initialize variables"]
    
    # Partial match: 1 out of 2
    payload = {"fix": "We should add a break condition to the loop"}
    score = grader.grade("suggest_fix", payload, [], expected_fixes)
    assert score == 0.5
    
    # Full match: 2 out of 2
    payload = {"fix": "Add a break condition and initialize variables"}
    score = grader.grade("suggest_fix", payload, [], expected_fixes)
    assert score == 1.0

def test_data_grader_remove_null():
    grader = DataGrader()
    dataset = {
        "rows": [
            {"id": 1, "val": 0.5},
            {"id": 2, "val": None}
        ]
    }
    assert grader.grade(dataset, "remove_null") == 0.0
    
    dataset_clean = {
        "rows": [
            {"id": 1, "val": 0.5},
            {"id": 2, "val": 0.8}
        ]
    }
    assert grader.grade(dataset_clean, "remove_null") == 1.0

def test_data_grader_normalize():
    grader = DataGrader()
    dataset = {
        "rows": [{"val": 10.0}]
    }
    assert grader.grade(dataset, "normalize") == 0.0
    
    dataset_norm = {
        "rows": [{"val": 0.5}]
    }
    assert grader.grade(dataset_norm, "normalize") == 1.0

def test_data_grader_fix_schema():
    grader = DataGrader()
    dataset = {
        "required_keys": ["id", "name"],
        "rows": [{"id": 1}]
    }
    # Some rows missing keys
    assert grader.grade(dataset, "fix_schema") == 0.5
    
    dataset_fixed = {
        "required_keys": ["id", "name"],
        "rows": [{"id": 1, "name": "test"}]
    }
    assert grader.grade(dataset_fixed, "fix_schema") == 1.0
