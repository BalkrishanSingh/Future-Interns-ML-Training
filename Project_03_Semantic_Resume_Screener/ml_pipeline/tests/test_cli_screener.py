"""Tests for CLI Screener (ml_pipeline/cli_screener.py).

Verifies PII scrubbing, text extraction, composite score calculation,
table output format, top-N ranking, and skill audit correctness.
"""

from pathlib import Path
import subprocess
import sys
import pytest
from pypdf import PdfWriter


@pytest.fixture
def sample_resumes_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample resume text files."""
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    # Resume 1: Alice - Strong Python and SQL, Masters, 5 years exp
    (resumes_dir / "resume_alice.txt").write_text(
        "Alice Smith\n"
        "Contact: alice@example.com, +1-555-123-4567, https://alice.dev\n"
        "Senior Software Engineer with 5 years of experience in Python, SQL, and FastAPI.\n"
        "Education: Masters in Computer Science\n",
        encoding="utf-8",
    )

    # Resume 2: Bob - Junior dev with React and JavaScript, Bachelors, 1 year exp
    (resumes_dir / "resume_bob.txt").write_text(
        "Bob Jones\n"
        "Email: bob.jones@mail.org | Phone: (555) 987-6543 | https://bobcodes.io\n"
        "Junior Web Developer with 1 year of experience building React and JavaScript apps.\n"
        "Education: Bachelors in Information Technology\n",
        encoding="utf-8",
    )

    # Resume 3: Charlie - DevOps Engineer with Docker, Kubernetes, Linux, 4 years exp
    (resumes_dir / "resume_charlie.txt").write_text(
        "Charlie Brown\n"
        "charlie@infra.net | 555-234-5678\n"
        "DevOps Engineer with 4 years experience in Linux, Docker, Kubernetes, and CI/CD.\n"
        "Education: Bachelors of Science in Engineering\n",
        encoding="utf-8",
    )

    # Resume 4: Dana - Data Scientist with Python, PyTorch, SQL, PhD, 6 years exp
    (resumes_dir / "resume_dana.txt").write_text(
        "Dana Scully\n"
        "dana.scully@research.edu | https://dana-lab.org\n"
        "Lead Data Scientist with 6 years experience in Python, PyTorch, and SQL.\n"
        "Education: PhD in Computational Science\n",
        encoding="utf-8",
    )

    return resumes_dir


def test_output_format(sample_resumes_dir: Path):
    """Assert stdout contains ranked table headers and at least 1 data row."""
    cmd = [
        sys.executable,
        "-m",
        "ml_pipeline.cli_screener",
        "--jd",
        "Software Engineer with Python and SQL",
        "--resumes",
        str(sample_resumes_dir),
        "--top",
        "3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = proc.stdout

    # Assert headers are present
    assert "Rank" in stdout
    assert "Candidate" in stdout
    assert "Score" in stdout
    assert "Matched Skills" in stdout
    assert "Missing Skills" in stdout

    # Assert table contains at least 1 data row with a candidate name
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    data_rows = [line for line in lines if line.startswith("1") or ("|" in line and "resume_" in line)]
    assert len(data_rows) >= 1, f"Expected at least 1 data row, got:\n{stdout}"


def test_top_n_ranking(sample_resumes_dir: Path):
    """Assert only --top N rows are printed."""
    top_n = 2
    cmd = [
        sys.executable,
        "-m",
        "ml_pipeline.cli_screener",
        "--jd",
        "Software Engineer with Python, SQL, and Docker",
        "--resumes",
        str(sample_resumes_dir),
        "--top",
        str(top_n),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = proc.stdout

    # Parse data rows by finding lines starting with numeric ranks (1, 2, ...)
    rows = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append(parts)

    assert len(rows) == top_n, f"Expected exactly {top_n} rows, but got {len(rows)}:\n{stdout}"
    assert rows[0][0] == "1"
    assert rows[1][0] == "2"

    # Verify descending score order
    score1 = float(rows[0][2])
    score2 = float(rows[1][2])
    assert score1 >= score2, f"Expected descending scores: {score1} >= {score2}"


def test_skill_audit_correctness(tmp_path: Path):
    """Given a fixture resume with known skills and a fixture JD, assert matched/missing counts are correct."""
    from ml_pipeline.cli_screener import compute_skill_audit, extract_skills_from_jd, extract_skills_from_resume

    jd_text = "Job Title: Backend Developer. Required Skills: Python, SQL, Docker, Kubernetes. Minimum Experience: 3 years."
    resume_text = (
        "Experienced Backend Developer with 4 years in Python and SQL.\n"
        "Passionate about building scalable web microservices."
    )

    required_skills = extract_skills_from_jd(jd_text)
    candidate_skills = extract_skills_from_resume(resume_text, required_skills)
    audit = compute_skill_audit(required_skills, candidate_skills)

    # Python and SQL should be matched
    matched_lower = {s.lower() for s in audit["matched"]}
    missing_lower = {s.lower() for s in audit["missing"]}

    assert "python" in matched_lower
    assert "sql" in matched_lower
    assert len(audit["matched"]) == 2

    # Docker and Kubernetes should be missing
    assert "docker" in missing_lower
    assert "kubernetes" in missing_lower
    assert len(audit["missing"]) == 2


def test_pii_scrubbing():
    """Verify regex correctly scrubs email, phone numbers, and URLs."""
    from ml_pipeline.cli_screener import scrub_pii

    raw = (
        "Alice Smith\n"
        "Email: alice.smith@example.co.uk\n"
        "Phone: +1 (555) 123-4567 or 555.987.6543\n"
        "Portfolio: https://alice-smith.github.io/portfolio\n"
        "Skills: Python, SQL"
    )
    scrubbed = scrub_pii(raw)

    assert "[REDACTED_EMAIL]" in scrubbed
    assert "alice.smith@example.co.uk" not in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "555" not in scrubbed
    assert "[REDACTED_URL]" in scrubbed
    assert "https://" not in scrubbed
    assert "Python, SQL" in scrubbed


def test_pdf_extraction(tmp_path: Path):
    """Verify extracting text from PDF files using pypdf."""
    from ml_pipeline.cli_screener import extract_text_from_file

    pdf_path = tmp_path / "test_resume.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # We can test plain text extraction or reading
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Should not raise exception
    text = extract_text_from_file(pdf_path)
    assert isinstance(text, str)


def test_composite_score_formula():
    """Verify composite scoring formula matches spec §4.5 (W_skills=0.5, W_exp=0.3, W_edu=0.2)."""
    import numpy as np
    from ml_pipeline.cli_screener import (
        CandidateProfile,
        JobSpec,
        compute_composite_score,
    )

    # Case:
    # Cosine = 1.0 (non-neg) -> 0.70 * 100 = 70
    # Coverage_req = 2/3 (skills: req=[Python, SQL, Docker], cand=[Python, SQL]) -> 0.30 * (2/3) * 100 = 20
    # S_skills = 70 + 20 = 90
    # Exp: actual 2, req 4 -> 50%
    # Edu: Masters -> 85
    # Composite: 0.5 * 90 + 0.3 * 50 + 0.2 * 85 = 45 + 15 + 17 = 77.0
    profile = CandidateProfile(
        candidate_id="cand-001",
        technical_skills=["Python", "SQL"],
        total_years_experience=2.0,
        highest_degree="Masters",
        embedding=np.array([1.0, 0.0]),
    )
    job_spec = JobSpec(
        title="Python Engineer",
        required_skills=["Python", "SQL", "Docker"],
        min_experience_years=4.0,
        embedding=np.array([1.0, 0.0]),
    )
    scores = compute_composite_score(
        profile=profile,
        job_spec=job_spec,
    )
    assert pytest.approx(scores["s_skills"], 0.1) == 90.0
    assert pytest.approx(scores["s_exp"], 0.1) == 50.0
    assert pytest.approx(scores["s_edu"], 0.1) == 85.0
    assert pytest.approx(scores["composite_score"], 0.1) == 77.0


def test_semantic_skill_matching_and_recall_coverage():
    """Verify ML matches Machine Learning, and coverage evaluates to 100%."""
    import numpy as np
    from ml_pipeline.cli_screener import (
        CandidateProfile,
        JobSpec,
        compute_composite_score,
        compute_skill_audit,
    )

    # Candidate with "Machine Learning" when required is "ML"
    audit = compute_skill_audit(
        required_skills=["ML", "Python"],
        candidate_skills=["Machine Learning", "Python"],
    )
    assert len(audit["matched"]) == 2
    assert len(audit["missing"]) == 0
    assert len(audit["bonus"]) == 0  # "Machine Learning" is consumed, not left in bonus

    profile = CandidateProfile(
        candidate_id="cand-synonym",
        technical_skills=["Machine Learning", "Python"],
        total_years_experience=3.0,
        highest_degree="Bachelors",
        embedding=np.array([1.0, 0.0]),
    )
    job_spec = JobSpec(
        title="AI Engineer",
        required_skills=["ML", "Python"],
        min_experience_years=3.0,
        embedding=np.array([1.0, 0.0]),
    )
    scores = compute_composite_score(profile, job_spec)
    # Cosine = 1.0 (70), Recall Coverage = 2/2 = 1.0 (30) -> S_skills = 100.0
    # S_exp = 100.0, S_edu = 70.0 -> Composite = 0.5*100 + 0.3*100 + 0.2*70 = 94.0
    assert pytest.approx(scores["s_skills"], 0.1) == 100.0
    assert pytest.approx(scores["composite_score"], 0.1) == 94.0


def test_composite_score_with_dataclasses():
    """Verify compute_composite_score using bundled CandidateProfile and JobSpec dataclasses."""
    import numpy as np
    from ml_pipeline.cli_screener import (
        CandidateProfile,
        JobSpec,
        compute_composite_score,
        extract_skills_from_job_spec,
    )

    jd_text = "Senior Python Engineer. Required Skills: Python, SQL, Docker. Minimum Experience: 4 years."
    skills = extract_skills_from_job_spec(jd_text)
    assert "Python" in skills or "python" in [s.lower() for s in skills]

    job_spec = JobSpec(
        title="Senior Python Engineer",
        required_skills=["Python", "SQL", "Docker"],
        min_experience_years=4.0,
        embedding=np.array([1.0, 0.0]),
    )
    profile = CandidateProfile(
        candidate_id="cand-001",
        technical_skills=["Python", "SQL"],
        total_years_experience=2.0,
        highest_degree="Masters",
        embedding=np.array([1.0, 0.0]),
    )

    scores = compute_composite_score(
        profile=profile,
        job_spec=job_spec,
    )
    assert pytest.approx(scores["s_skills"], 0.1) == 90.0
    assert pytest.approx(scores["s_exp"], 0.1) == 50.0
    assert pytest.approx(scores["s_edu"], 0.1) == 85.0
    assert pytest.approx(scores["composite_score"], 0.1) == 77.0


def test_recall_coverage_not_penalizing_bonus_skills():
    """Verify that candidates with many bonus skills are not penalized."""
    import numpy as np
    from ml_pipeline.cli_screener import (
        CandidateProfile,
        JobSpec,
        compute_composite_score,
        compute_skill_audit,
    )

    job_spec = JobSpec(
        title="Quality & Test Engineer",
        required_skills=["Python", "SQL", "Docker"],
        min_experience_years=3.0,
        embedding=np.array([1.0, 0.0]),
    )
    # Candidate matches all 3 required skills, plus 11 bonus skills
    candidate = CandidateProfile(
        candidate_id="ENGINEERING_12011623",
        technical_skills=[
            "Python", "SQL", "Docker",
            "C#", "C++", "Java", "Go", "Rust", "Kubernetes", "AWS", "Terraform", "Linux", "Kafka", "Redis"
        ],
        total_years_experience=3.0,
        highest_degree="Bachelors",
        embedding=np.array([1.0, 0.0]),
    )
    audit = compute_skill_audit(job_spec.required_skills, candidate.technical_skills)
    assert len(audit["matched"]) == 3
    assert len(audit["missing"]) == 0
    assert len(audit["bonus"]) >= 11

    scores = compute_composite_score(candidate, job_spec)
    # Under Pure Recall Coverage, 3 of 3 required matched gives coverage 1.0 (30 pts)
    # With Cosine 1.0 (70 pts), S_skills is 100.0, NOT diluted by the 11 bonus skills!
    assert pytest.approx(scores["s_skills"], 0.1) == 100.0


def test_preferred_skill_weighting_and_audit():
    """Verify additive preferred skill weighting and audit breakdowns."""
    import numpy as np
    from ml_pipeline.cli_screener import (
        CandidateProfile,
        JobSpec,
        compute_composite_score,
        compute_skill_audit,
    )

    job_spec = JobSpec(
        title="Full Stack Engineer",
        required_skills=["Python", "FastAPI"],
        preferred_skills=["Docker", "Redis"],
        min_experience_years=4.0,
        embedding=np.array([1.0, 0.0]),
    )

    # Candidate A: 100% required matched (2/2), 0% preferred matched (0/2)
    # Plus unconsumed candidate skill "Kubernetes"
    cand_a = CandidateProfile(
        candidate_id="cand-a",
        technical_skills=["Python", "FastAPI", "Kubernetes"],
        total_years_experience=4.0,
        highest_degree="Bachelors",
        embedding=np.array([1.0, 0.0]),
    )
    audit_a = compute_skill_audit(
        job_spec.required_skills,
        cand_a.technical_skills,
        preferred_skills=job_spec.preferred_skills,
    )
    assert set(audit_a["matched"]) == {"Python", "FastAPI"}
    assert audit_a["missing"] == []
    assert audit_a["bonus"] == ["Kubernetes"]

    # Coverage_req = 1.0, Coverage_pref = 0.0
    # Additive boost: SkillMatch = min(1.0, 1.0 + 0.15 * 0.0) = 1.0 (never penalized for missing preferred skills!)
    # S_skills = (0.70 * 1.0 + 0.30 * 1.0) * 100 = 100.0
    scores_a = compute_composite_score(cand_a, job_spec)
    assert pytest.approx(scores_a["s_skills"], 0.1) == 100.0

    # Candidate B: 50% required matched (1/2 Python), 100% preferred matched (2/2 Docker, Redis)
    cand_b = CandidateProfile(
        candidate_id="cand-b",
        technical_skills=["Python", "Docker", "Redis"],
        total_years_experience=4.0,
        highest_degree="Bachelors",
        embedding=np.array([1.0, 0.0]),
    )
    audit_b = compute_skill_audit(
        job_spec.required_skills,
        cand_b.technical_skills,
        preferred_skills=job_spec.preferred_skills,
    )
    # Matched includes both required and preferred, with subcategories exposed
    assert set(audit_b["matched"]) == {"Python", "Docker", "Redis"}
    assert audit_b["matched_required"] == ["Python"]
    assert set(audit_b["matched_preferred"]) == {"Docker", "Redis"}
    assert audit_b["missing"] == ["FastAPI"]
    assert audit_b["bonus"] == []

    # Coverage_req = 0.5, Coverage_pref = 1.0
    # SkillMatch = min(1.0, 0.5 + 0.15 * 1.0) = 0.65
    # S_skills = (0.70 * 1.0 + 0.30 * 0.65) * 100 = 89.5
    scores_b = compute_composite_score(cand_b, job_spec)
    assert pytest.approx(scores_b["s_skills"], 0.1) == 89.5

    # Candidate C: 100% required matched (2/2) and 100% preferred matched (2/2)
    cand_c = CandidateProfile(
        candidate_id="cand-c",
        technical_skills=["Python", "FastAPI", "Docker", "Redis"],
        total_years_experience=4.0,
        highest_degree="Bachelors",
        embedding=np.array([1.0, 0.0]),
    )
    audit_c = compute_skill_audit(
        job_spec.required_skills,
        cand_c.technical_skills,
        preferred_skills=job_spec.preferred_skills,
    )
    assert set(audit_c["matched"]) == {"Python", "FastAPI", "Docker", "Redis"}
    assert audit_c["missing"] == []
    assert audit_c["bonus"] == []

    scores_c = compute_composite_score(cand_c, job_spec)
    assert pytest.approx(scores_c["s_skills"], 0.1) == 100.0


def test_synonym_skills_not_duplicated_or_in_bonus():
    """Verify ML and Machine Learning do not count as two distinct skills or leak into bonus."""
    from ml_pipeline.cli_screener import compute_skill_audit, canonicalize_skills

    canonical = canonicalize_skills(["ML", "Machine Learning", "Python"], preferred_keywords=["ML"])
    assert "Machine Learning" not in canonical
    assert "ML" in canonical
    assert len(canonical) == 2

    audit = compute_skill_audit(
        required_skills=["ML"],
        candidate_skills=["ML", "Machine Learning", "Python"],
    )
    assert audit["matched"] == ["ML"]
    assert audit["missing"] == []
    assert "Machine Learning" not in audit["bonus"]
    assert audit["bonus"] == ["Python"]
