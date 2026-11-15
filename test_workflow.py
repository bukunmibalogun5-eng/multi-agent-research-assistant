
from workflow import assess_research_coverage

def test_all_sub_questions_fully_covered():
    research_results = [
        {
            "sub_question": "What is machine learning?",
            "status": "success",
            "findings": [{"claim": "Finding 1"}, {"claim": "Finding 2"}],
        },
        {
            "sub_question": "How does supervised learning work?",
            "status": "success",
            "findings": [{"claim": "Finding 3"}],
        },
    ]

    assessment, percentage = assess_research_coverage(research_results)

    assert percentage == 100.0
    assert len(assessment) == 2
    assert all(
        item["coverage_status"] == "covered"
        for item in assessment
    )


def test_mixed_coverage_statuses():
    research_results = [
        {
            "sub_question": "Question 1",
            "status": "success",
            "findings": [{"claim": "Finding 1"}],
        },
        {
            "sub_question": "Question 2",
            "status": "partial_success",
            "findings": [{"claim": "Finding 2"}],
        },
        {
            "sub_question": "Question 3",
            "status": "insufficient_evidence",
            "findings": [],
        },
        {
            "sub_question": "Question 4",
            "status": "research_failed",
            "findings": [],
        },
    ]

    assessment, percentage = assess_research_coverage(research_results)

    assert percentage == 25.0
    assert [item["coverage_status"] for item in assessment] == [
        "covered",
        "partially_covered",
        "insufficient_evidence",
        "not_covered",
    ]


def test_empty_research_results():
    assessment, percentage = assess_research_coverage([])

    assert assessment == []
    assert percentage == 0


def test_success_with_no_findings_is_not_covered():
    research_results = [
        {
            "sub_question": "Question 1",
            "status": "success",
            "findings": [],
        }
    ]

    assessment, percentage = assess_research_coverage(research_results)

    assert assessment[0]["coverage_status"] == "insufficient_evidence"
    assert percentage == 0


def test_partial_success_with_no_findings_is_not_partially_covered():
    research_results = [
        {
            "sub_question": "Question 1",
            "status": "partial_success",
            "findings": [],
        }
    ]

    assessment, percentage = assess_research_coverage(research_results)

    assert assessment[0]["coverage_status"] == "insufficient_evidence"
    assert percentage == 0


def test_partial_success_with_findings_is_partially_covered():
    research_results = [
        {
            "sub_question": "Question 1",
            "status": "partial_success",
            "findings": [{"claim": "A validated finding"}],
        }
    ]

    assessment, percentage = assess_research_coverage(research_results)

    assert assessment[0]["coverage_status"] == "partially_covered"
    assert percentage == 0
