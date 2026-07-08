"""Generate synthetic Job Descriptions across 24 resume categories.

Supports seed context from custom_job_descriptions.csv and optional Gemini Flash
generation with seamless fallback to deterministic realistic specifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional

CATEGORIES: list[str] = [
    "INFORMATION-TECHNOLOGY",
    "BUSINESS-DEVELOPMENT",
    "ADVOCATE",
    "CHEF",
    "FINANCE",
    "ENGINEERING",
    "ACCOUNTANT",
    "FITNESS",
    "AVIATION",
    "SALES",
    "HEALTHCARE",
    "CONSULTANT",
    "BANKING",
    "CONSTRUCTION",
    "PUBLIC-RELATIONS",
    "HR",
    "DESIGNER",
    "ARTS",
    "TEACHER",
    "APPAREL",
    "DIGITAL-MEDIA",
    "AGRICULTURE",
    "AUTOMOBILE",
    "BPO",
]

CSV_HEADERS: list[str] = [
    "Category",
    "title",
    "required_skills",
    "preferred_skills",
    "min_experience_years",
]

# Rich deterministic fallback job descriptions covering all 24 categories
DETERMINISTIC_JDS: dict[str, list[dict[str, str | int]]] = {
    "INFORMATION-TECHNOLOGY": [
        {
            "Category": "INFORMATION-TECHNOLOGY",
            "title": "Full Stack Software Engineer",
            "required_skills": "Python|JavaScript|React|PostgreSQL|REST APIs",
            "preferred_skills": "Docker|AWS|TypeScript|CI/CD",
            "min_experience_years": 3,
        },
        {
            "Category": "INFORMATION-TECHNOLOGY",
            "title": "Cloud DevOps Engineer",
            "required_skills": "Linux|Kubernetes|Docker|Terraform|CI/CD",
            "preferred_skills": "AWS|GCP|Prometheus|Python",
            "min_experience_years": 4,
        },
        {
            "Category": "INFORMATION-TECHNOLOGY",
            "title": "Data Platform Engineer",
            "required_skills": "SQL|Python|Apache Spark|Airflow|Data Warehousing",
            "preferred_skills": "Kafka|Snowflake|dbt|GCP",
            "min_experience_years": 4,
        },
    ],
    "BUSINESS-DEVELOPMENT": [
        {
            "Category": "BUSINESS-DEVELOPMENT",
            "title": "Business Development Manager",
            "required_skills": "B2B Sales|Lead Generation|Pipeline Management|Client Relations",
            "preferred_skills": "CRM|Market Research|Contract Negotiation|SaaS",
            "min_experience_years": 4,
        },
        {
            "Category": "BUSINESS-DEVELOPMENT",
            "title": "Strategic Partnerships Director",
            "required_skills": "Partnership Development|Deal Structuring|Cross-functional Leadership|Revenue Strategy",
            "preferred_skills": "Executive Presentations|Financial Modeling|Venture Deals",
            "min_experience_years": 6,
        },
        {
            "Category": "BUSINESS-DEVELOPMENT",
            "title": "Commercial Growth Specialist",
            "required_skills": "Outreach Strategy|Account Mapping|Competitive Intelligence|Proposal Writing",
            "preferred_skills": "HubSpot|Enterprise Pitching|Sales Enablement",
            "min_experience_years": 3,
        },
    ],
    "ADVOCATE": [
        {
            "Category": "ADVOCATE",
            "title": "Corporate Legal Counsel",
            "required_skills": "Contract Drafting|Regulatory Compliance|Legal Research|Corporate Governance",
            "preferred_skills": "M&A Advisory|IP Licensing|Commercial Litigation",
            "min_experience_years": 5,
        },
        {
            "Category": "ADVOCATE",
            "title": "Litigation Associate",
            "required_skills": "Case Briefing|Trial Preparation|Deposition Experience|Legal Drafting",
            "preferred_skills": "Dispute Resolution|Arbitration|Court Advocacy",
            "min_experience_years": 3,
        },
        {
            "Category": "ADVOCATE",
            "title": "Regulatory Compliance Attorney",
            "required_skills": "Statutory Compliance|Policy Drafting|Risk Assessment|Internal Investigations",
            "preferred_skills": "Data Privacy Law|Anti-Trust|Labor Law Advisory",
            "min_experience_years": 4,
        },
    ],
    "CHEF": [
        {
            "Category": "CHEF",
            "title": "Executive Chef",
            "required_skills": "Menu Development|Kitchen Management|Food Costing|Culinary Arts",
            "preferred_skills": "HACCP Certification|Fine Dining|Inventory Management",
            "min_experience_years": 7,
        },
        {
            "Category": "CHEF",
            "title": "Sous Chef",
            "required_skills": "Line Cooking|Food Preparation|Staff Supervision|Kitchen Safety",
            "preferred_skills": "Pastry Techniques|Banquet Prep|Portion Control",
            "min_experience_years": 3,
        },
        {
            "Category": "CHEF",
            "title": "Head Pastry Chef & Baker",
            "required_skills": "Artisan Baking|Pastry Production|Dessert Styling|Sanitation Protocols",
            "preferred_skills": "Chocolate Work|Recipe Scaling|Bakery Inventory",
            "min_experience_years": 4,
        },
    ],
    "FINANCE": [
        {
            "Category": "FINANCE",
            "title": "Financial Analyst",
            "required_skills": "Financial Modeling|Valuation|Excel|Variance Analysis",
            "preferred_skills": "Power BI|SQL|CFA Candidate|Budgeting",
            "min_experience_years": 3,
        },
        {
            "Category": "FINANCE",
            "title": "Portfolio Risk Manager",
            "required_skills": "Risk Assessment|Quantitative Analysis|Asset Allocation|Portfolio Theory",
            "preferred_skills": "Python|Bloomberg Terminal|Fixed Income|Derivatives",
            "min_experience_years": 5,
        },
        {
            "Category": "FINANCE",
            "title": "Corporate Treasury Analyst",
            "required_skills": "Cash Flow Forecasting|Liquidity Management|FX Hedging|Bank Relations",
            "preferred_skills": "Treasury Management Systems|Credit Analysis|ERP",
            "min_experience_years": 3,
        },
    ],
    "ENGINEERING": [
        {
            "Category": "ENGINEERING",
            "title": "Mechanical Design Engineer",
            "required_skills": "CAD|SolidWorks|FEA Analysis|Thermal Dynamics|Product Design",
            "preferred_skills": "GD&T|Rapid Prototyping|Manufacturing Methods",
            "min_experience_years": 4,
        },
        {
            "Category": "ENGINEERING",
            "title": "Systems Engineering Specialist",
            "required_skills": "System Architecture|Requirements Engineering|Risk Analysis|Integration Testing",
            "preferred_skills": "MATLAB|Simulink|ISO Standards|Embedded Systems",
            "min_experience_years": 5,
        },
        {
            "Category": "ENGINEERING",
            "title": "Electrical Hardware Engineer",
            "required_skills": "PCB Design|Circuit Analysis|Schematic Capture|Oscilloscope Testing",
            "preferred_skills": "Altium Designer|EMI/EMC Compliance|FPGA",
            "min_experience_years": 3,
        },
    ],
    "ACCOUNTANT": [
        {
            "Category": "ACCOUNTANT",
            "title": "Senior Accountant",
            "required_skills": "General Ledger|Month-end Close|GAAP Compliance|Account Reconciliation",
            "preferred_skills": "CPA|SAP|QuickBooks|Financial Reporting",
            "min_experience_years": 4,
        },
        {
            "Category": "ACCOUNTANT",
            "title": "Audit Associate",
            "required_skills": "Internal Controls|Audit Procedures|Financial Statements|Workpaper Preparation",
            "preferred_skills": "SOX Compliance|ERP Systems|Tax Compliance",
            "min_experience_years": 2,
        },
        {
            "Category": "ACCOUNTANT",
            "title": "Tax & Compliance Specialist",
            "required_skills": "Corporate Tax Filing|Direct & Indirect Tax|Tax Audits|Tax Planning",
            "preferred_skills": "GST/VAT|Transfer Pricing|Tax Provisioning",
            "min_experience_years": 4,
        },
    ],
    "FITNESS": [
        {
            "Category": "FITNESS",
            "title": "Fitness Director & Lead Trainer",
            "required_skills": "Personal Training|Strength & Conditioning|Program Design|Client Assessment",
            "preferred_skills": "CPT Certification|Nutrition Coaching|Group Fitness Leadership",
            "min_experience_years": 4,
        },
        {
            "Category": "FITNESS",
            "title": "Exercise Physiologist",
            "required_skills": "Biomechanical Analysis|Cardiac Rehab|Functional Movement Screen|Injury Prevention",
            "preferred_skills": "ACSM Certification|Sports Science|Athletic Training",
            "min_experience_years": 3,
        },
        {
            "Category": "FITNESS",
            "title": "Wellness & Conditioning Coach",
            "required_skills": "Endurance Training|Mobility Screening|Client Goal Setting|Fitness Technology",
            "preferred_skills": "First Aid/CPR|Wearables Tracking|Group Coaching",
            "min_experience_years": 2,
        },
    ],
    "AVIATION": [
        {
            "Category": "AVIATION",
            "title": "Commercial Airline Pilot",
            "required_skills": "ATP License|Flight Navigation|Multi-Engine Rating|Crew Resource Management",
            "preferred_skills": "Boeing/Airbus Type Rating|Flight Simulator Instructor|FAA Regulations",
            "min_experience_years": 6,
        },
        {
            "Category": "AVIATION",
            "title": "Flight Operations Manager",
            "required_skills": "Flight Dispatch|Aviation Safety Management|Fleet Scheduling|Regulatory Compliance",
            "preferred_skills": "ICAO Standards|Meteorology Analysis|Emergency Response Planning",
            "min_experience_years": 5,
        },
        {
            "Category": "AVIATION",
            "title": "Avionics Maintenance Technician",
            "required_skills": "Avionics Troubleshooting|Aircraft Inspection|A&P Certification|Wiring Schematics",
            "preferred_skills": "Flight Radar Systems|FAA Part 145|Line Maintenance",
            "min_experience_years": 3,
        },
    ],
    "SALES": [
        {
            "Category": "SALES",
            "title": "Enterprise Account Executive",
            "required_skills": "SaaS Sales|Complex Deal Negotiation|Pipeline Forecasting|Stakeholder Management",
            "preferred_skills": "Salesforce CRM|MEDDPICC|Outbound Prospecting",
            "min_experience_years": 5,
        },
        {
            "Category": "SALES",
            "title": "Inside Sales Representative",
            "required_skills": "Cold Calling|Lead Qualification|Product Demos|CRM Hygiene",
            "preferred_skills": "HubSpot|Email Campaigns|Objection Handling",
            "min_experience_years": 2,
        },
        {
            "Category": "SALES",
            "title": "Regional Sales Director",
            "required_skills": "Territory Management|Quota Achievement|Sales Team Leadership|Client Retention",
            "preferred_skills": "Channel Sales|Executive Selling|Contract Structuring",
            "min_experience_years": 7,
        },
    ],
    "HEALTHCARE": [
        {
            "Category": "HEALTHCARE",
            "title": "Clinical Nurse Specialist",
            "required_skills": "Patient Assessment|BLS/ACLS|Clinical Documentation|Evidence-Based Practice",
            "preferred_skills": "Epic EMR|Specialty Board Certification|Triage Protocol",
            "min_experience_years": 4,
        },
        {
            "Category": "HEALTHCARE",
            "title": "Healthcare Operations Manager",
            "required_skills": "Clinical Workflow Optimization|HIPAA Compliance|Staff Scheduling|Patient Safety",
            "preferred_skills": "Hospital Administration|JCAHO Standards|Health Informatics",
            "min_experience_years": 5,
        },
        {
            "Category": "HEALTHCARE",
            "title": "Medical Laboratory Scientist",
            "required_skills": "Specimen Analysis|Diagnostic Testing|Quality Control|Biohazard Safety",
            "preferred_skills": "Hematology|LIS Systems|Microbiology Protocols",
            "min_experience_years": 3,
        },
    ],
    "CONSULTANT": [
        {
            "Category": "CONSULTANT",
            "title": "Management Consultant",
            "required_skills": "Strategic Advisory|Market Entry Strategy|Stakeholder Interviewing|Executive Presentation",
            "preferred_skills": "Quantitative Modeling|Process Optimization|Change Management",
            "min_experience_years": 4,
        },
        {
            "Category": "CONSULTANT",
            "title": "Technology Strategy Consultant",
            "required_skills": "Digital Transformation|IT Architecture Assessment|Roadmap Planning|Vendor Selection",
            "preferred_skills": "Cloud Migration|Enterprise Agile|Business Architecture",
            "min_experience_years": 5,
        },
        {
            "Category": "CONSULTANT",
            "title": "Operations & Supply Chain Consultant",
            "required_skills": "Process Mapping|Cost Reduction Analysis|Lean Six Sigma|Capacity Planning",
            "preferred_skills": "Simulation Modeling|ERP Advisory|Procurement Transformation",
            "min_experience_years": 4,
        },
    ],
    "BANKING": [
        {
            "Category": "BANKING",
            "title": "Commercial Relationship Manager",
            "required_skills": "Credit Underwriting|Loan Structuring|Financial Statement Analysis|Client Prospecting",
            "preferred_skills": "Treasury Management|Syndicated Loans|Regulatory Compliance",
            "min_experience_years": 4,
        },
        {
            "Category": "BANKING",
            "title": "Credit Risk Analyst",
            "required_skills": "Credit Scoring|Portfolio Risk Analysis|Default Modeling|Stress Testing",
            "preferred_skills": "Moody's Analytics|SQL|Basel III Regulations",
            "min_experience_years": 3,
        },
        {
            "Category": "BANKING",
            "title": "Branch Operations Manager",
            "required_skills": "Retail Banking Operations|KYC/AML Compliance|Customer Service Excellence|Audit Readiness",
            "preferred_skills": "Core Banking Platforms|Cash Management|Branch Sales Coaching",
            "min_experience_years": 5,
        },
    ],
    "CONSTRUCTION": [
        {
            "Category": "CONSTRUCTION",
            "title": "Construction Project Manager",
            "required_skills": "Subcontractor Oversight|Project Scheduling|Budget Control|Site Safety Management",
            "preferred_skills": "Procore|OSHA 30|AutoCAD|Cost Estimation",
            "min_experience_years": 5,
        },
        {
            "Category": "CONSTRUCTION",
            "title": "Civil Site Engineer",
            "required_skills": "Site Inspection|Quality Assurance|Structural Specifications|Blueprint Reading",
            "preferred_skills": "Civil 3D|Surveying|Environmental Compliance",
            "min_experience_years": 3,
        },
        {
            "Category": "CONSTRUCTION",
            "title": "Construction Estimator & Quantity Surveyor",
            "required_skills": "Cost Estimation|Quantity Takeoff|Bid Preparation|Material Procurement",
            "preferred_skills": "Planswift|BIM Modeling|Value Engineering",
            "min_experience_years": 4,
        },
    ],
    "PUBLIC-RELATIONS": [
        {
            "Category": "PUBLIC-RELATIONS",
            "title": "Public Relations Manager",
            "required_skills": "Media Relations|Press Releases|Crisis Communication|Campaign Strategy",
            "preferred_skills": "Brand Storytelling|Media Tracking Tools|Influencer Relations",
            "min_experience_years": 4,
        },
        {
            "Category": "PUBLIC-RELATIONS",
            "title": "Corporate Communications Lead",
            "required_skills": "Internal Communications|Executive Speechwriting|Reputation Management|Content Strategy",
            "preferred_skills": "Social Media PR|Stakeholder Messaging|Event PR",
            "min_experience_years": 5,
        },
        {
            "Category": "PUBLIC-RELATIONS",
            "title": "Media Outreach Coordinator",
            "required_skills": "Journalist Pitching|Press Kit Assembly|Coverage Clipping|Media Lists",
            "preferred_skills": "Cision|Muck Rack|Social Monitoring",
            "min_experience_years": 2,
        },
    ],
    "HR": [
        {
            "Category": "HR",
            "title": "Technical Recruiter",
            "required_skills": "Technical Sourcing|Candidate Screening|Applicant Tracking Systems|Offer Negotiation",
            "preferred_skills": "LinkedIn Recruiter|Boolean Search|Employer Branding",
            "min_experience_years": 3,
        },
        {
            "Category": "HR",
            "title": "Human Resources Business Partner",
            "required_skills": "Employee Relations|Talent Management|Organizational Design|Labor Law Compliance",
            "preferred_skills": "SHRM-CP|Performance Coaching|HR Analytics",
            "min_experience_years": 5,
        },
        {
            "Category": "HR",
            "title": "Total Rewards & Compensation Analyst",
            "required_skills": "Salary Benchmarking|Benefits Administration|Job Evaluation|HRIS Management",
            "preferred_skills": "Workday|Equity Compensation|FLSA Compliance",
            "min_experience_years": 3,
        },
    ],
    "DESIGNER": [
        {
            "Category": "DESIGNER",
            "title": "Senior UI/UX Designer",
            "required_skills": "Wireframing|Figma|Design Systems|User Research|Usability Testing",
            "preferred_skills": "Interactive Prototyping|Information Architecture|Micro-interactions",
            "min_experience_years": 4,
        },
        {
            "Category": "DESIGNER",
            "title": "Product Designer",
            "required_skills": "User-Centered Design|Design Thinking|UI Component Libraries|Rapid Prototyping",
            "preferred_skills": "Accessibility Standards|UX Writing|Mobile App UX",
            "min_experience_years": 3,
        },
        {
            "Category": "DESIGNER",
            "title": "Brand & Visual Identity Designer",
            "required_skills": "Typography|Vector Illustration|Brand Style Guides|Adobe Creative Suite",
            "preferred_skills": "Motion Graphics|Packaging Design|Design Systems",
            "min_experience_years": 3,
        },
    ],
    "ARTS": [
        {
            "Category": "ARTS",
            "title": "Art Director",
            "required_skills": "Creative Direction|Visual Storytelling|Brand Identity|Team Mentorship",
            "preferred_skills": "Adobe Creative Cloud|Campaign Concepts|Print & Digital Production",
            "min_experience_years": 6,
        },
        {
            "Category": "ARTS",
            "title": "Fine Arts Curator",
            "required_skills": "Exhibition Planning|Artwork Provenance|Art History Analysis|Catalog Writing",
            "preferred_skills": "Gallery Management|Grant Writing|Conservation Techniques",
            "min_experience_years": 4,
        },
        {
            "Category": "ARTS",
            "title": "Museum Exhibition Designer",
            "required_skills": "Spatial Design|Exhibition Graphics|3D Modeling|Artifact Preservation",
            "preferred_skills": "Vectorworks|Visitor Experience Mapping|Lighting Design",
            "min_experience_years": 3,
        },
    ],
    "TEACHER": [
        {
            "Category": "TEACHER",
            "title": "High School STEM Teacher",
            "required_skills": "Curriculum Delivery|Lesson Planning|Classroom Management|Formative Assessment",
            "preferred_skills": "State Teaching Certification|Differentiated Instruction|EdTech Tools",
            "min_experience_years": 3,
        },
        {
            "Category": "TEACHER",
            "title": "Instructional Specialist",
            "required_skills": "Pedagogical Coaching|Curriculum Design|Standardized Testing Analysis|Teacher Training",
            "preferred_skills": "Master of Education|Special Education Strategies|Blended Learning",
            "min_experience_years": 5,
        },
        {
            "Category": "TEACHER",
            "title": "Primary Education Educator",
            "required_skills": "Early Childhood Pedagogy|Literacy Instruction|Parent Engagement|Behavior Management",
            "preferred_skills": "Montessori Methodology|Individualized Education Plans (IEP)|Child Psychology",
            "min_experience_years": 2,
        },
    ],
    "APPAREL": [
        {
            "Category": "APPAREL",
            "title": "Senior Apparel Designer",
            "required_skills": "Garment Construction|Tech Packs|Fabric Sourcing|Trend Forecasting",
            "preferred_skills": "Illustrator Fashion CAD|Fit Approvals|Costing Sheets",
            "min_experience_years": 5,
        },
        {
            "Category": "APPAREL",
            "title": "Fashion Production Merchandiser",
            "required_skills": "Vendor Negotiation|Sample Approval|Production Timeline Management|Quality Control",
            "preferred_skills": "Textile Science|Supply Chain Management|Inventory Planning",
            "min_experience_years": 3,
        },
        {
            "Category": "APPAREL",
            "title": "Textile Sourcing Specialist",
            "required_skills": "Yarn & Fabric Testing|Sustainability Standards|Mill Sourcing|Dyeing & Finishing",
            "preferred_skills": "Circular Fashion|Raw Material Costing|Global Sourcing",
            "min_experience_years": 4,
        },
    ],
    "DIGITAL-MEDIA": [
        {
            "Category": "DIGITAL-MEDIA",
            "title": "Digital Marketing Strategist",
            "required_skills": "SEO/SEM Strategy|Google Analytics|Paid Ad Campaigns|Content Marketing",
            "preferred_skills": "HubSpot|Performance Marketing|A/B Testing",
            "min_experience_years": 3,
        },
        {
            "Category": "DIGITAL-MEDIA",
            "title": "Social Media & Content Lead",
            "required_skills": "Social Media Strategy|Short-form Video Production|Community Engagement|Copywriting",
            "preferred_skills": "Creator Partnerships|TikTok/Instagram Analytics|Canva/Premiere",
            "min_experience_years": 3,
        },
        {
            "Category": "DIGITAL-MEDIA",
            "title": "Multimedia Content Producer",
            "required_skills": "Video Editing|Motion Graphics|Audio Post-Production|Storyboarding",
            "preferred_skills": "Premiere Pro|After Effects|Podcast Production",
            "min_experience_years": 3,
        },
    ],
    "AGRICULTURE": [
        {
            "Category": "AGRICULTURE",
            "title": "Agronomist & Crop Specialist",
            "required_skills": "Crop Yield Optimization|Soil Fertility Management|Pest Scouting|Irrigation Planning",
            "preferred_skills": "Precision Agriculture|GPS/GIS Mapping|Fertilizer Formulations",
            "min_experience_years": 4,
        },
        {
            "Category": "AGRICULTURE",
            "title": "Farm Operations Manager",
            "required_skills": "Agricultural Equipment Management|Crop Scheduling|Harvest Logistics|Labor Supervision",
            "preferred_skills": "Organic Certification Standards|Supply Chain Logistics|Budget Planning",
            "min_experience_years": 5,
        },
        {
            "Category": "AGRICULTURE",
            "title": "Precision Agriculture Technologist",
            "required_skills": "Drone Field Mapping|Telemetry Sensor Analysis|AgTech IoT|Variable Rate Technology",
            "preferred_skills": "GIS Software|Data Analytics|Smart Irrigation Automation",
            "min_experience_years": 3,
        },
    ],
    "AUTOMOBILE": [
        {
            "Category": "AUTOMOBILE",
            "title": "Automotive Systems Engineer",
            "required_skills": "Powertrain Diagnostics|CAN Bus Architecture|ECU Calibration|Vehicle Dynamics",
            "preferred_skills": "MATLAB/Simulink|EV Battery Systems|ISO 26262 Functional Safety",
            "min_experience_years": 4,
        },
        {
            "Category": "AUTOMOBILE",
            "title": "Automotive Service & Fleet Supervisor",
            "required_skills": "Fleet Maintenance Operations|OBD-II Diagnostics|Shop Safety Standards|Technician Supervision",
            "preferred_skills": "ASE Certification|Warranty Processing|Inventory Management",
            "min_experience_years": 5,
        },
        {
            "Category": "AUTOMOBILE",
            "title": "Electric Vehicle Diagnostic Specialist",
            "required_skills": "High Voltage Safety|Battery Management Systems (BMS)|Inverter Troubleshooting|Wiring Diagrams",
            "preferred_skills": "Thermal Management|EV Telematics|Oscilloscope Analysis",
            "min_experience_years": 3,
        },
    ],
    "BPO": [
        {
            "Category": "BPO",
            "title": "BPO Operations Team Lead",
            "required_skills": "SLA Management|Quality Assurance Auditing|Agent Coaching|KPI Tracking",
            "preferred_skills": "Call Center Workforce Management|Six Sigma Yellow Belt|Zendesk/Salesforce",
            "min_experience_years": 3,
        },
        {
            "Category": "BPO",
            "title": "Customer Experience Operations Manager",
            "required_skills": "Omnichannel Support Operations|Process Optimization|CSAT/NPS Improvement|Team Leadership",
            "preferred_skills": "Vendor Management|Contact Center Technology|Capacity Planning",
            "min_experience_years": 5,
        },
        {
            "Category": "BPO",
            "title": "Workforce Management & Quality Analyst",
            "required_skills": "Call Volume Forecasting|Agent Scheduling|Real-time Queue Adherence|Speech Analytics",
            "preferred_skills": "Verint|NICE WFM|Root Cause Analysis",
            "min_experience_years": 3,
        },
    ],
}


def _resolve_path(path_str: str, default_filename: str) -> Path:
    """Resolve file path against CWD or repo root."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    # Try looking in repo root
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / path_str
    if candidate.exists():
        return candidate
    return p


def load_seed_jds(seed_path: str | Path) -> list[dict[str, str]]:
    """Load hand-written seed JDs from CSV if available."""
    resolved = _resolve_path(str(seed_path), "custom_job_descriptions.csv")
    if not resolved.exists():
        return []

    seeds: list[dict[str, str]] = []
    with open(resolved, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Category") and row.get("title"):
                seeds.append({
                    "Category": row["Category"].strip(),
                    "title": row["title"].strip(),
                    "required_skills": row.get("required_skills", "").strip(),
                    "preferred_skills": row.get("preferred_skills", "").strip(),
                    "min_experience_years": row.get("min_experience_years", "2").strip(),
                })
    return seeds


def _fetch_from_gemini(category: str, needed: int, api_key: str) -> list[dict[str, str | int]]:
    """Attempt to generate job specifications via Gemini Flash REST endpoint."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Generate exactly {needed} realistic job specifications for the category: {category}.\n"
        "Return a valid JSON array of objects with the exact schema:\n"
        "[\n"
        "  {\n"
        f'    "Category": "{category}",\n'
        '    "title": "Specific Job Title",\n'
        '    "required_skills": "skill1|skill2|skill3",\n'
        '    "preferred_skills": "skill4|skill5",\n'
        '    "min_experience_years": 3\n'
        "  }\n"
        "]\n"
        "Do not wrap in markdown quotes. Just return the JSON array."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        data = json.loads(response.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            valid_items = []
            for item in parsed:
                if isinstance(item, dict) and item.get("title"):
                    valid_items.append({
                        "Category": category,
                        "title": str(item["title"]),
                        "required_skills": str(item.get("required_skills", "")),
                        "preferred_skills": str(item.get("preferred_skills", "")),
                        "min_experience_years": int(item.get("min_experience_years", 3)),
                    })
            return valid_items
    return []


def generate_jds(
    output_path: str | Path = "generated_jds.csv",
    seed_path: str | Path = "custom_job_descriptions.csv",
    target_per_category: int = 2,
    use_gemini: bool = True,
) -> list[dict[str, str | int]]:
    """Generate job descriptions ensuring all 24 categories have at least target_per_category JDs.

    Args:
        output_path: Path to write/append generated_jds.csv.
        seed_path: Path to custom_job_descriptions.csv.
        target_per_category: Minimum JDs required per category (default: 2).
        use_gemini: Whether to try Gemini Flash if GEMINI_API_KEY is available.

    Returns:
        List of all job description dicts present in output_path.
    """
    output_file = _resolve_path(str(output_path), "generated_jds.csv")

    existing_rows: list[dict[str, str | int]] = []
    category_counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    # 1. Load existing generated JDs if file exists
    if output_file.exists():
        with open(output_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("Category", "").strip()
                if cat:
                    existing_rows.append({
                        "Category": cat,
                        "title": row.get("title", "").strip(),
                        "required_skills": row.get("required_skills", "").strip(),
                        "preferred_skills": row.get("preferred_skills", "").strip(),
                        "min_experience_years": int(row.get("min_experience_years", 2)),
                    })
                    category_counts[cat] = category_counts.get(cat, 0) + 1

    # 2. Check seed file if existing output is missing categories
    seed_jds = load_seed_jds(seed_path)
    if seed_jds:
        # Incorporate seeds for categories that need more rows
        for s in seed_jds:
            cat = s["Category"]
            if cat in category_counts and category_counts[cat] < target_per_category:
                # Check for duplicate title
                if not any(r["Category"] == cat and r["title"] == s["title"] for r in existing_rows):
                    existing_rows.append({
                        "Category": cat,
                        "title": s["title"],
                        "required_skills": s["required_skills"],
                        "preferred_skills": s["preferred_skills"],
                        "min_experience_years": int(s["min_experience_years"]),
                    })
                    category_counts[cat] += 1

    # 3. For any category with fewer than target_per_category rows, generate realistic JDs
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() if use_gemini else ""

    for category in CATEGORIES:
        current_count = category_counts.get(category, 0)
        needed = target_per_category - current_count
        if needed <= 0:
            # Idempotent skip: category already has >= target_per_category rows
            continue

        generated_for_cat: list[dict[str, str | int]] = []

        # Try Gemini API if key is present
        if gemini_key:
            try:
                gemini_results = _fetch_from_gemini(category, needed, gemini_key)
                if len(gemini_results) >= needed:
                    generated_for_cat.extend(gemini_results[:needed])
            except Exception:
                # Seamless fallback on any failure
                generated_for_cat = []

        # If Gemini was not used or failed/returned insufficient, use deterministic fallback
        if len(generated_for_cat) < needed:
            fallback_options = DETERMINISTIC_JDS.get(category, [])
            for item in fallback_options:
                if len(generated_for_cat) >= needed:
                    break
                if not any(r["Category"] == category and r["title"] == item["title"] for r in existing_rows + generated_for_cat):
                    generated_for_cat.append(dict(item))

        # Add newly generated JDs to main list
        existing_rows.extend(generated_for_cat)
        category_counts[category] = category_counts.get(category, 0) + len(generated_for_cat)

        # Log progress to stdout 
        print(f"Category {category}: generated {len(generated_for_cat)} JDs")

    # 4. Save to output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({
                "Category": r["Category"],
                "title": r["title"],
                "required_skills": r["required_skills"],
                "preferred_skills": r["preferred_skills"],
                "min_experience_years": r["min_experience_years"],
            })

    return existing_rows


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate synthetic JDs for 24 categories.")
    parser.add_argument("--output", default="generated_jds.csv", help="Path to output CSV")
    parser.add_argument("--seed", default="custom_job_descriptions.csv", help="Path to seed CSV")
    parser.add_argument("--target", type=int, default=2, help="Target JDs per category (default: 2)")
    args = parser.parse_args()

    results = generate_jds(output_path=args.output, seed_path=args.seed, target_per_category=args.target)
    print(f"Total JDs generated / available: {len(results)} across {len(CATEGORIES)} categories.")


if __name__ == "__main__":
    main()
